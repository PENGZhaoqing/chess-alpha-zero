"""
Encapsulates the worker which evaluates newly-trained models and picks the best one
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from multiprocessing import Manager
from time import sleep
from time import time

from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib.data_helper import get_next_generation_model_dirs, pretty_print
from chess_zero.lib.model_helper import save_as_best_model, load_best_model_weight

logger = getLogger(__name__)


def start(config: Config):
    return EvaluateWorker(config).start()

class EvaluateWorker:
    """
    Worker which evaluates trained models and keeps track of the best one

    Attributes:
        :ivar Config config: config to use for evaluation
        :ivar PlayConfig config: PlayConfig to use to determine how to play, taken from config.eval.play_config
        :ivar ChessModel current_model: currently chosen best model
        :ivar Manager m: multiprocessing manager
        :ivar list(Connection) cur_pipes: pipes on which the current best ChessModel is listening which will be used to
            make predictions while playing a game.
    """
    def __init__(self, config: Config):
        """
        :param config: Config to use to control how evaluation should work
        """
        self.config = config
        self.play_config = config.eval.play_config
        self.current_model = self.load_current_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.current_model.get_pipes(self.play_config.search_threads) for _ in range(self.play_config.max_processes)])

    def start(self):
        """
        Start evaluation, endlessly loading the latest models from the directory which stores them and
        checking if they do better than the current model, saving the result in self.current_model
        """
        start_time = time()
        while True:
            ng_model, model_dir = self.load_next_generation_model()

            ng_model = self.load_current_model()

            logger.debug(f"start evaluate model {model_dir}")
            ng_is_great = self.evaluate_model(ng_model)
            if ng_is_great:
                logger.debug(f"New Model become best model: {model_dir}")
            print(f"time={time() - start_time:5.1f}s ")
            exit()
            #     save_as_best_model(ng_model)
            #     self.current_model = ng_model
            # self.move_model(model_dir)

    def evaluate_model(self, ng_model):
        """
        Given a model, evaluates it by playing a bunch of games against the current model.

        :param ChessModel ng_model: model to evaluate
        :return: true iff this model is better than the current_model
        """
        ng_pipes = self.m.list([ng_model.get_pipes(self.play_config.search_threads) for _ in range(self.play_config.max_processes)])

        futures = []
        with ProcessPoolExecutor(max_workers=self.play_config.max_processes) as executor:
            for game_idx in range(self.config.eval.game_num):
                fut = executor.submit(play_game, self.config, cur=self.cur_pipes, ng=ng_pipes, current_white=(game_idx % 2 == 0))
                futures.append(fut)

            results = []
            f_total = 0.0
            f_total_step = 0.0
            for fut in as_completed(futures):
                # ng_score := if ng_model win -> 1, lose -> 0, draw -> 0.5
                ng_score, env, current_white,total,total_step = fut.result()
                results.append(ng_score)
                f_total+=total
                f_total_step+=total_step
                win_rate = sum(results) / len(results)
                game_idx = len(results)
                logger.debug(f"game {game_idx:3}: ng_score={ng_score:.1f} as {'black' if current_white else 'white'} "
                             f"{'by resign ' if env.resigned else '          '}"
                             f"win_rate={win_rate*100:5.1f}% "
                             f"{env.board.fen().split(' ')[0]}"
                             f"time={f_total:5.1f}s total step = {f_total_step} average = {f_total/f_total_step:5.3f}")

                colors = ("current_model", "ng_model")
                if not current_white:
                    colors = reversed(colors)
                pretty_print(env, colors)

                # if len(results)-sum(results) >= self.config.eval.game_num * (1-self.config.eval.replace_rate):
                #     logger.debug(f"lose count reach {results.count(0)} so give up challenge")
                #     return False
                # if sum(results) >= self.config.eval.game_num * self.config.eval.replace_rate:
                #     logger.debug(f"win count reach {results.count(1)} so change best model")
                #     return True

        win_rate = sum(results) / len(results)
        logger.debug(f"winning rate {win_rate*100:.1f}%")
        return win_rate >= self.config.eval.replace_rate

    def move_model(self, model_dir):
        """
        Moves the newest model to the specified directory

        :param file model_dir: directory where model should be moved
        """
        rc = self.config.resource
        new_dir = os.path.join(rc.next_generation_model_dir, "copies", model_dir.name)
        os.rename(model_dir, new_dir)

    def load_current_model(self):
        """
        Loads the best model from the standard directory.
        :return ChessModel: the model
        """
        model = ChessModel(self.config)
        load_best_model_weight(model)
        return model

    def load_next_generation_model(self):
        """
        Loads the next generation model from the standard directory
        :return (ChessModel, file): the model and the directory that it was in
        """
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)
            if dirs:
                break
            logger.info("There is no next generation model to evaluate")
            sleep(60)
        model_dir = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model = ChessModel(self.config)
        model.load(config_path, weight_path)
        return model, model_dir


def play_game(config, cur, ng, current_white: bool) -> (float, ChessEnv, bool):
    """
    Plays a game against models cur and ng and reports the results.

    :param Config config: config for how to play the game
    :param ChessModel cur: should be the current model
    :param ChessModel ng: should be the next generation model
    :param bool current_white: whether cur should play white or black
    :return (float, ChessEnv, bool): the score for the ng model
        (0 for loss, .5 for draw, 1 for win), the env after the game is finished, and a bool
        which is true iff cur played as white in that game.
    """
    cur_pipes = cur.pop()
    ng_pipes = ng.pop()
    env = ChessEnv().reset()

    current_player = ChessPlayer(config, pipes=cur_pipes, play_config=config.eval.play_config)
    ng_player = ChessPlayer(config, pipes=ng_pipes, play_config=config.eval.play_config)
    if current_white:
        white, black = current_player, ng_player
    else:
        white, black = ng_player, current_player

    start_time = time()
    total = 0.0
    total_step = 0.0
    
    while not env.done:
        if env.white_to_move:
            if(current_white):
                action = white.action(env)
            else:
                start_time = time()
                action = white.action_modify(env)
                total+=time() - start_time
                total_step+=1
        else:
            if(current_white):
                start_time = time()
                action = black.action_modify(env)
                total+=time() - start_time
                total_step+=1
            else:
                action = black.action(env)

        env.step(action)
        if env.num_halfmoves >= config.eval.max_game_length:
            env.adjudicate()

    #print(f"time={total:5.1f}s total step = {total_step} average = {total/total_step:5.1f}")
    #exit()

    if env.winner == Winner.draw:
        ng_score = 0.5
    elif env.white_won == current_white:
        ng_score = 0
    else:
        ng_score = 1
    cur.append(cur_pipes)
    ng.append(ng_pipes)
    return ng_score, env, current_white, total, total_step
