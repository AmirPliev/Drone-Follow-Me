class Agent_Type():
    PPO_Agent       = "ppo"
    DQN_Agent       = "dqn"
    HARD_Agent      = "hard"
    RANDOM_Agent    = "random"

class Reset_Method():
    World_Initial_State     = "OriginalPlace"
    Random_Around_Person    = "RandomPlaceAround"
    Directly_Behind_Person  = "DirectlyBehind"

class Environments():
    BlocksNormal        = "BlocksNormal"
    BlocksObstacles     = "BlocksObstacles"
    FactoryTest         = "Factory"
    FullList            = ["BlocksNormal", "BlocksObstacles", "Factory"]
