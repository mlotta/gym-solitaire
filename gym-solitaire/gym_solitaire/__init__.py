from gym.envs.registration import register

register(
    id='solitaire-v0',
    entry_point='gym_solitaire.envs:SolitaireEnv',
)
