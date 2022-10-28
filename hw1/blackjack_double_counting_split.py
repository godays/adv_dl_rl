import gym
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    return float(a > b) - float(a < b)

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

counting_rule = {
    1: -2,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 2,
    7: 1,
    8: 0,
    9: -1,
    10: -2
}


# def draw_card(np_random):
#     return int(np_random.choice(deck))


# def draw_hand(np_random):
#     return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackDoubleCountingSplitEnv(gym.Env):
    """Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards. (Virtually for all Blackjack games today).

    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).

    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.

    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.

    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Discrete(90),
            spaces.Discrete(4),
        ))
        self.seed()
        
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.count = 0
        self.split_status = 0
        self.reward = 0

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def draw_card(self, np_random):
        card = int(np_random.choice(self.deck))
        self.deck.remove(card)
        return card


    def draw_hand(self, np_random):
        return [self.draw_card(np_random), self.draw_card(np_random)]


    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit: add a card to players hand and return
            card = self.draw_card(self.np_random)
            self.count_card(card)
            self.player.append(card)
            if is_bust(self.player):
                done = self.request_done()
                self.reward += -1.
            else:
                done = False
                self.reward += 0.
        elif action == 0:  # stick: play out the dealers hand, and score
            done = self.request_done()
            self.fill_dealer()
            temp_reward = cmp(score(self.player), score(self.dealer))
            self.reward += temp_reward
            if self.natural and is_natural(self.player) and temp_reward == 1.:
                self.reward += 0.5
        elif action == 2: # double
            card = self.draw_card(self.np_random)
            self.count_card(card)
            self.player.append(card)
            done = self.request_done()
            self.fill_dealer()
            self.reward += cmp(score(self.player), score(self.dealer)) * 2
        elif action == 3: # split
            done = False
            assert self.split_status == 1, self._get_obs()
            self.player2 = [self.player.pop()]
            self.split_status = 2
            
        # If no split after first step then no split at all
        if self.split_status == 1:
            self.split_status = 0
            
        
        return self._get_obs(), self.get_reward(done), done, {}
    
    def fill_dealer(self):
        while sum_hand(self.dealer) < 17:
            card = self.draw_card(self.np_random)
            self.dealer.append(card)
            self.count_card(card)
    
    def request_done(self):
        if self.split_status == 2:
            self.split_status = 3
            self.player, self.player2 = self.player2, self.player
            return False
        else:
            return True
        
    def get_reward(self, done):
        if self.split_status in {0, 3} and done:
            return self.reward
        else:
            return 0

    def _get_obs(self):
        return (
            sum_hand(self.player), 
            self.dealer[0], 
            usable_ace(self.player), 
            self.count,
            self.split_status
        )
    
    def count_card(self, card):
        self.count += counting_rule[card] 
        

    def reset(self):
        if len(self.deck) < 15:
            self.deck = deck[:]
            self.count = 0
        
        self.reward = 0
        self.dealer = self.draw_hand(self.np_random)
        self.count_card(self.dealer[0])
        self.player = self.draw_hand(self.np_random)
        self.count_card(self.player[0])
        self.count_card(self.player[1])
        
        self.split_status = int(self.player[0] == self.player[1])
        return self._get_obs()
    
    def get_actions(self):
        if self.split_status == 1:
            return (0, 1, 2, 3)
        else:
            return (0, 1, 2)

