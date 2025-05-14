# **SecureFed**
A federated learning framework built with Flower and PyTorch to evaluate the robustness of FL systems under data poisoning attacks. The system uses the PathMNIST dataset from the MedMNIST benchmark, simulates realistic adversarial scenarios by flipping labels on malicious clients, and defends against poisoned updates using a trust and reputation-based aggregation strategy.

## **🚀Highlights**
- 🧠 **CNN-based Federated Learning**: Trains CNN models (e.g., ResNet18) in a non-IID federated setting.
- 🔄 **Adversarial Simulation**: Simulates data poisoning attacks by flipping labels on compromised clients.
- 🛡️ **Defense via Trust & Reputation**: Excludes unreliable clients using trust and reputation scoring.
- 📉 **Metric Logging**: Logs accuracy, loss, and trust scores; saves model checkpoints each round.
- ⚡ **GPU-Accelerated**: Optimized for CUDA-enabled training on GPUs.

## **▶️Running the Project**

### 1. **Requirements**:
The `requirements.txt` file contains all the necessary Python dependencies for the project. Place any additional required dependencies here. You can install them using:
``` bash
    pip install -r requirements.txt
```

### 2. **Run the Full Federated Learning Setup**
You can start both the server and multiple clients using the provided script:
```bash
./run.sh <number_of_clients>
```

For Attack simulation 
``` bash
nohup python3 -u main_client.py $i 1 > logs/client_$i.log 2>&1 &
```
By default, run.sh starts clients with attack_flag=1. Replace 1 with 0 to disable label-flipping attacks for clients. You can also customize the sleep duration or logging paths if needed.

## **🗂️Project Structure**
```bash
    │   main_client.py
    │   main_server.py
    │   plot.py
    │   pyproject.toml
    │   README.md
    │   requirements.txt
    │   run.sh
    │   
    ├───backend
    │      client_app.py
    │      distance.py
    │      model.py
    │      save_model_strategy.py
    │      task.py
    │
    ├───checkpoints
    │       model_round_1.pth
    │       model_round_2.pth
    │
    └───logs
            client_0.log
            client_1.log
            server.log
 ```

 ## **📊Results**

We evaluated the robustness of our federated learning setup across three scenarios:

---

### ✅ Scenario 1: No Attack (Standard FL)

- ⚙️ **Setup**: 10 clients, clean data, 10 rounds, 5 local epochs per round.
- 📈 **Accuracy**: Reached approximately **80%**.
- 📉 **Loss**: Decreased steadily and remained **below 2.0** throughout training.
- 🧠 **Insight**: The global model demonstrated stable convergence and strong performance under clean conditions.

---

### ❌ Scenario 2: With Attack (No Defense)

- 🧪 **Attack Simulation**: 3 out of 10 clients had **50% of their labels flipped** to simulate data poisoning.
- ⚠️ **Impact**:
  - First-round **loss** for malicious clients was **~11.52**, compared to **< 5** for clean clients.
  - The global model's **loss** gradually decreased but stabilized at **~2.2** after 10 rounds.
  - **Accuracy** only reached **~55–60%**, far below the clean setup.
- 🔎 **Insight**: Poisoned updates significantly slowed down convergence and reduced global performance.

---

### 🛡️ Scenario 3: With Attack + Trust & Reputation Defense

- 🔐 **Defense**:
  - Trust and reputation scores were computed in early rounds.
  - From **Round 4**, clients with low trust scores were excluded from aggregation.
- 📈 **Recovery**:
  - Global **accuracy** recovered to **~70%** by Round 10.
  - Model began to converge more reliably after filtering out malicious contributions.
- ✅ **Insight**: The defense mechanism improved accuracy and mitigated attack effects, though it didn’t fully match the clean scenario.

---

### 📌 Summary Table

| Scenario              | Accuracy     | Loss     | Remarks                                |
|-----------------------|--------------|----------|----------------------------------------|
| No Attack             | ~80%         | < 2.0    | Smooth convergence                      |
| Attack, No Defense    | ~55–60%      | ~2.2     | High instability, slow convergence      |
| Attack + Defense      | ~70%         | ~2.0     | Partial recovery, improved robustness   |

📄 **[Read the full report](./report/securefed_report.pdf)** for methodology, experiments and results.