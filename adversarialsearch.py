import numpy as np
from typing import Callable

from adversarialsearchproblem import (
    Action,
    AdversarialSearchProblem,
    State as GameState,
)


def minimax(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the minimax algorithm on ASPs, assuming that the given game is
    both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
    """
    state = asp.get_start_state()
    player = state.player_to_move()
    util_val_1, move = max_value(asp, state, player)
    return move

def max_value(asp, state, player):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    value = -np.inf
    for a in asp.get_available_actions(state):
        util_val_2, action_2 = min_value(asp, asp.transition(state, a), player)
        if util_val_2 > value:
            value, move = util_val_2, a
    return value, move

def min_value(asp, state, player):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    value = np.Inf
    for a in asp.get_available_actions(state):
        util_val_2, action_2 = max_value(asp, asp.transition(state, a), player)
        if util_val_2 < value:
            value, move = util_val_2, a
    return value, move
    
def alpha_beta(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    state = asp.get_start_state()
    player = state.player_to_move()
    util_val_1, move = ab_max_value(asp, state, player, -np.inf, np.inf)
    return move

def ab_max_value(asp, state, player, alpha, beta):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    value = -np.inf
    for a in asp.get_available_actions(state):
        util_val_2, action_2 = ab_min_value(asp, asp.transition(state, a), player, alpha, beta)
        if util_val_2 > value:
            value, move = util_val_2, a
            alpha = max(alpha, value)
            if value >= beta:
                return value, move
    return value, move

def ab_min_value(asp, state, player, alpha, beta):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    value = np.inf
    for a in asp.get_available_actions(state):
        util_val_2, action_2 = ab_max_value(asp, asp.transition(state, a), player, alpha, beta)
        if util_val_2 < value:
            value, move = util_val_2, a
            beta = min(beta, value)
            if value <= alpha:
                return value, move
    return value, move


def alpha_beta_cutoff(
    asp: AdversarialSearchProblem[GameState, Action],
    cutoff_ply: int,
    # See AdversarialSearchProblem:heuristic_func
    heuristic_func: Callable[[GameState], float],
) -> Action:
    """
    This function should:
    - search through the asp using alpha-beta pruning
    - cut off the search after cutoff_ply moves have been made.

    Input:
        asp - an AdversarialSearchProblem
        cutoff_ply - an Integer that determines when to cutoff the search and
            use heuristic_func. For example, when cutoff_ply = 1, use
            heuristic_func to evaluate states that result from your first move.
            When cutoff_ply = 2, use heuristic_func to evaluate states that
            result from your opponent's first move. When cutoff_ply = 3 use
            heuristic_func to evaluate the states that result from your second
            move. You may assume that cutoff_ply > 0.
        heuristic_func - a function that takes in a GameState and outputs a
            real number indicating how good that state is for the player who is
            using alpha_beta_cutoff to choose their action. You do not need to
            implement this function, as it should be provided by whomever is
            calling alpha_beta_cutoff, however you are welcome to write
            evaluation functions to test your implemention. The heuristic_func
            we provide does not handle terminal states, so evaluate terminal
            states the same way you evaluated them in the previous algorithms.
    Output:
        an action(an element of asp.get_available_actions(asp.get_start_state()))
    """

    state = asp.get_start_state()
    player = state.player_to_move()
    util_val_1, move = abp_max_value(asp, state, -np.inf, np.inf, cutoff_ply, player)
    return move

def abp_max_value(asp, state, alpha, beta, depth, player):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    elif depth == 0:
        return asp.heuristic_func(state,player), None
    depth -= 1
    for a in asp.get_available_actions(state):
        var, m = abp_min_value(asp, asp.transition(state, a), alpha, beta, depth, player)
        alpha = max(alpha, var)
        move = a
        if alpha >= beta:
            return beta, None
    return alpha, move

def abp_min_value(asp, state, alpha, beta, depth, player):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    elif depth == 0:
        return asp.heuristic_func(state,player), None
    depth -= 1
    for a in asp.get_available_actions(state):
        var, m = abp_max_value(asp, asp.transition(state, a), alpha, beta, depth, player)
        beta = min(beta, var)
        move = a
        if beta <= alpha:
            return alpha, None
    return beta, move