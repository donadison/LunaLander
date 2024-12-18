import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from MoonLanderEnvGym_copy import MoonLanderEnv  # Import Twojego środowiska
from stable_baselines3.dqn.policies import DQNPolicy  # Zmiana na odpowiednią lokalizację

# Tworzenie środowiska
env = MoonLanderEnv()
env = Monitor(env)  # Dodanie monitora do środowiska (zapisywanie wyników)

# Opcjonalnie, stworzenie wektoryzowanego środowiska (dla większej stabilności)
vec_env = DummyVecEnv([lambda: env])

# Konfiguracja loggera (opcjonalnie, do zapisu wyników)
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# Zdefiniowanie niestandardowej architektury sieci
class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Definicja niestandardowej sieci (256x256)
        self.net_arch = [256, 256]  # Zwiększenie rozmiaru sieci

    def _build_mlp_extractor(self):
        """
        Zbuduj niestandardową sieć MLP (Multi-Layer Perceptron).
        """
        return nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.features_dim)
        )

# Inicjalizacja modelu DQN z niestandardową polityką
model = DQN(
    CustomDQNPolicy,  # Użycie niestandardowej polityki
    vec_env,          # Wektoryzowane środowisko
    learning_rate=1e-3,  # Szybkość uczenia
    buffer_size=50000,   # Rozmiar replay buffer
    learning_starts=1000,  # Początek uczenia po X krokach
    batch_size=64,       # Wielkość batcha
    gamma=0.99,          # Współczynnik dyskontowania
    target_update_interval=500,  # Co ile kroków aktualizacja target network
    exploration_fraction=0.1,    # Frakcja eksploracji
    exploration_final_eps=0.05,  # Minimalne epsilon (eksploracja)
    verbose=1,          # Szczegółowe logi
    tensorboard_log=log_dir  # Ścieżka do logów TensorBoard
)

# Callback do zapisu checkpointów
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # Co ile kroków zapisywać model
    save_path="./models/",
    name_prefix="dqn_moonlander"
)

# Trening modelu
model.learn(
    total_timesteps=100000,  # Liczba kroków treningowych
    callback=checkpoint_callback,
    progress_bar=True
)

# Ocena modelu
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
print(f"Średnia nagroda: {mean_reward} +/- {std_reward}")

# Zapis wytrenowanego modelu
model.save("dqn_moonlander")
