# Aircraft Electrical System Health Monitor: Technical Interview Preparation Guide

This document prepares you to defend this project in an Electrical Engineering / Software Engineering undergraduate interview. Every answer is technically honest, grounded in the actual codebase, and defensible without overclaiming.

---

## Quick Defense Reference (Elevator Pitch & Direct Answers)

### Did you work on a real aircraft or certified flight hardware?
> **Answer:** "No. This project is a Python-based software prototype and educational simulation. Because real aircraft electrical telemetry is proprietary and restricted by aviation security standards (ITAR/DO-178C), I built a simulation grounded in representative aircraft electrical engineering principles (115V AC 400Hz and 28V DC). It does not interface with certified avionics hardware or safety-critical flight computers."

### Where did your dataset come from?
> **Answer:** "I generated the time-series data programmatically using Python and NumPy. The generator implements physical characteristics such as generator regulation droop, load-proportional current draw, thermal/harmonic drift, and Gaussian measurement noise across discrete operating regimes (low load, normal cruise, high load, transients). I then injected controlled mathematical faults with known ground-truth timestamps."

### How did you achieve your accuracy / F1 score?
> **Answer:** "I evaluated the trained system against a dedicated 600-second operational dataset containing 10 injected fault events. Using scikit-learn metrics, I compared model predictions with the uncorrupted ground-truth labels. The combined system achieved **90.57% accuracy, 72.88% precision, 65.94% recall, and 69.24% F1-score**, with a **100% event detection success rate** and an average detection latency of **0.360 seconds**."

---

## Part 1: Basic & Systems Questions

### 1. What is this project?
An educational, condition-based monitoring prototype for aircraft electrical distribution buses (115V AC 400Hz and 28V DC). It simulates sensor time series (voltage, current, frequency, power), injects 10 controlled fault modes, monitors thresholds, detects multivariate anomalies using Isolation Forest, and generates health reports and diagnostic visualizations.

### 2. Why did you build it?
To explore how data-driven anomaly detection and classical limit checking can be integrated for predictive electrical maintenance. Aircraft electrical systems are safety-critical; early detection of degradation (such as generator diode wear or loose connections) before an hard threshold trip can prevent in-flight electrical disconnects.

### 3. Why did you use synthetic data?
Real aircraft flight data recorder (FDR) and line-replaceable unit (LRU) electrical bus telemetry are proprietary, tightly guarded by airframe manufacturers, and subject to defense/export restrictions. Synthetic data enabled rigorous, repeatable testing with exact ground-truth labels for fault injection.

### 4. Why Python?
Python is the standard language for scientific computing, prototyping, and rapid algorithm iteration. Its rich ecosystem (NumPy, SciPy, Pandas, scikit-learn, Matplotlib) allows end-to-end integration from physics simulation to machine learning and reporting.

### 5. Why Pandas?
Pandas provides structured DataFrame indexing, rolling window calculations (moving average, rolling variance), datetime/timestamp alignment, and structured data handling necessary for time-series feature engineering.

### 6. Why NumPy?
NumPy provides vectorized linear algebra, high-performance array arithmetic, and deterministic pseudo-random number generation for realistic signal synthesis (droop, noise, harmonic oscillation).

### 7. Why Matplotlib & Seaborn?
They provide publication-quality, deterministic rendering of multi-panel engineering dashboards, time-series telemetry plots, shaded event spans, and confusion matrices with full control over axes, ticks, and layouts without external web dependencies.

### 8. Why scikit-learn?
It is the industry standard for classical machine learning, providing robust implementations of unsupervised anomaly detection (`IsolationForest`), feature scaling (`StandardScaler`), and rigorous evaluation metrics (`precision_score`, `recall_score`, `confusion_matrix`).

### 9. What is Isolation Forest?
An unsupervised ensemble algorithm that isolates anomalies rather than profiling normal data clusters. It builds random binary trees by selecting a random feature and a random split value. Because anomalies are few and structurally distinct, they are isolated close to tree roots (short average path lengths).

### 10. Why not use a deep neural network (e.g., LSTM, Autoencoder)?
1. **Explainability & Certification Trajectory:** Avionics systems favor deterministic, explainable, lower-complexity algorithms over "black box" neural nets.
2. **Computational Footprint:** Isolation Forest trains in sub-seconds and performs inference with minimal memory and CPU overhead, suitable for low-power edge computers.
3. **Data Requirements:** Deep networks require thousands of flight hours of diverse training data to avoid catastrophic overfitting or hallucinated states.

---

## Part 2: Electrical Engineering Fundamentals

### 11. Why monitor voltage?
Voltage is the primary indicator of bus power quality and generator regulation. Under- or over-voltage can damage sensitive avionics, disconnect bus tie contactors, or cause line contactor drops.

### 12. Why monitor current?
Current indicates load demand, feeder stress, and fault conditions. High current signifies overloads, motor stalls, or short circuits; asymmetrical or spiking current indicates insulation breakdown or intermittent arcing.

### 13. Why monitor frequency?
In fixed-frequency AC systems (400 Hz), frequency is directly tied to generator shaft speed (constant speed drive / integrated drive generator). Frequency deviations indicate governor failure, mechanical drive hunting, or severe turbine speed fluctuations.

### 14. What does 115 V AC mean?
115 V RMS (Root Mean Square), representing the effective equivalent DC heating voltage of a sinusoidal AC waveform with a peak value of $115 \times \sqrt{2} \approx 162.6\text{ V}$.

### 15. Why is 400 Hz relevant to the simulation?
400 Hz is the standard frequency in aviation (MIL-STD-704). Higher frequency allows magnetic components (transformers, motors, generator cores) to be significantly smaller and lighter for the same power rating than 50/60 Hz utility equipment.

### 16. What is a 28 V DC bus?
28 V DC is the historic standard voltage for aircraft direct current systems, batteries, and essential avionics buses, derived from transformer-rectifier units (TRUs) or dedicated starter-generators and NiCd/Li-ion batteries.

### 17. What is overvoltage?
A sustained increase in bus voltage above the maximum continuous operating limit (e.g., >122 V RMS for AC, >30 V for DC), caused by generator control unit (GCU) voltage regulator failure, load dump, or feeder open-circuits.

### 18. What is undervoltage?
A sustained drop in bus voltage below minimum operating limit (e.g., <108 V RMS for AC, <25 V for DC), caused by generator overload, engine flameout, TRU failure, or excessive cable impedance.

### 19. What is overcurrent?
Current draw exceeding the continuous thermal rating of the bus or feeder conductors, threatening thermal damage, wire insulation melting, or circuit breaker trips.

### 20. What is a voltage sag?
A brief, transient decrease in voltage (lasting milliseconds to seconds), typically caused by high inrush current from starting heavy inductive loads like hydraulic pump motors.

### 21. What is a voltage spike?
A brief, steep transient overvoltage pulse, typically caused by inductive load switching, contactor opening, or atmospheric electrostatic discharge.

### 22. What can frequency deviation indicate?
Constant Speed Drive (CSD) or Integrated Drive Generator (IDG) mechanical wear, governor valve sticking, hydraulic differential failure, or sudden massive electrical shedding.

### 23. Why is sensor dropout different from an electrical fault?
An electrical fault is a physical energy anomaly on the bus; a sensor dropout (missing telemetry, NaN, disconnected lead) is an instrumentation failure. Conflating sensor loss with an electrical blackout causes erroneous bus transfer switching and unnecessary pilot workload.

---

## Part 3: Machine Learning & Anomaly Detection

### 24. How does Isolation Forest work?
It builds an ensemble of $N$ isolation trees (iTrees). Each iTree recursively subdivides data by randomly picking a feature and a random split between that feature's minimum and maximum. Anomalies sit in sparse regions and require few splits (low path length $h(x)$). The anomaly score is:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
where $E(h(x))$ is the average path length across all trees and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree of $n$ points.

### 25. What does contamination mean?
The proportion of outliers expected in the training dataset. It determines the decision threshold for the binary flag. In our semi-supervised design, we set contamination to 0.05 (5%) to establish a tight normal boundary on baseline data.

### 26. How did you select features?
Features were chosen based on electrical domain principles:
- Instantaneous readings: `voltage`, `current`, `frequency`
- Deviations: `voltage_deviation` ($V - V_{\text{nominal}}$), `frequency_deviation` ($f - f_{\text{nominal}}$)
- Dynamics: finite difference rates of change (`voltage_rate`, `current_rate`)
- Moving statistics: rolling mean and rolling standard deviation over 0.5s and 2.0s windows
- Interaction: `apparent_power` ($V \times I$)

### 27. Did you use labeled data to train the model?
No. The Isolation Forest was trained strictly in an unsupervised manner using a pure baseline normal operating dataset. Labels were reserved exclusively for post-inference evaluation.

### 28. How did you prevent data leakage?
Ground-truth columns (`true_fault`, `fault_type`, `fault_id`, `severity`) were completely isolated from the feature matrix `X`. Furthermore, feature scaling (`StandardScaler`) was fitted purely on baseline training data and applied to evaluation datasets without target statistics.

### 29. How did you evaluate the model?
By aligning the binary prediction `final_anomaly` against the binary ground-truth column `true_fault` across 6,000 samples, computing accuracy, precision, recall, F1, and confusion matrix, supplemented by fault-specific breakdown and event-level latency metrics.

### 30. Why is accuracy insufficient?
Because anomalies constitute a minority class (typically <10% of operational data). A naive dummy detector predicting "normal" for every sample would register 90%+ accuracy while detecting zero real failures.

### 31. What are precision and recall?
- **Precision:** $\frac{TP}{TP + FP}$ — proportion of flagged anomalies that were true electrical faults (avoids false alarms).
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — proportion of true electrical faults that were successfully detected (avoids missed faults).

### 32. What is a false positive?
A normal operating condition (e.g., motor inrush during transient loading) incorrectly flagged as an anomaly. High false positives cause "alarm fatigue" and unnecessary maintenance groundings.

### 33. What is a false negative?
A genuine electrical fault or degradation missed by the system. In aviation, false negatives are high-risk because unmitigated degradation can cascade into bus failure.

### 34. How did you handle class imbalance?
1. Evaluated using F1-score (harmonic mean of precision and recall) rather than accuracy alone.
2. Evaluated event-level detection (whether each distinct fault event was captured).
3. Used rolling temporal aggregation to prevent transient noise from distorting health assessments.

---

## Part 4: Project-Specific Architecture & Defense

### 35. How did you generate synthetic data?
Using a multi-state operating condition generator that transitions between low-load, normal-load, high-load, and transient regimes. The generator superimposes generator regulation droop, low-frequency sinusoidal thermal drift, and Gaussian measurement noise onto nominal setpoints (115V / 28V).

### 36. How did you inject faults?
By selecting non-overlapping time slots across the dataset and applying specific mathematical transformations to target slices: scaling voltage for over/undervoltage, multiplying current for overcurrent, adding frequency offsets for frequency deviation, inserting brief 2-8 sample surges for spikes/sags, setting sensor channels to NaN for dropouts, applying linear ramps for drift, and coupling simultaneous voltage-drop/current-surge for combined faults.

### 37. How did you know whether a fault was detected?
By checking if `final_anomaly == 1` within the interval $[t_{\text{start}}, t_{\text{end}}]$ of the fault event recorded in the fault catalog.

### 38. How did you calculate the reported detection percentage?
Sample-level detection rate is calculated as:
$$\text{Detection Rate} = \frac{\text{Detected Fault Samples}}{\text{Total Injected Fault Samples}} \times 100\%$$
Event-level detection is:
$$\text{Event Recall} = \frac{\text{Events with } \ge 1 \text{ detection}}{\text{Total Injected Events}} \times 100\% = \frac{10}{10} = 100.0\%$$

### 39. What was your biggest limitation?
The synthetic data simplifies complex non-linear electrical dynamics (such as sub-transient generator reactance, harmonic distortion from switching power supplies, and thermal derating). Furthermore, Isolation Forest does not natively model temporal sequencing like an HMM or state-space observer.

### 40. How would you adapt this to real aircraft sensor data?
1. Extract flight data recorder (ARINC 429 or MIL-STD-1553 bus) telemetry.
2. Implement adaptive baseline normalization conditioned on flight phase (Takeoff, Climb, Cruise, Descent).
3. Filter out expected transient electrical switching (e.g., fuel boost pump activation).
4. Run validation on historical line maintenance fault logs.

### 41. What changes would be required for real-time monitoring?
1. Transition from batch Pandas operations to fixed-size rolling ring buffers (e.g., 50-sample FIFO).
2. Compute moving statistics in constant time $O(1)$ using Welford's algorithm for rolling mean and variance.
3. Optimize Isolation Forest inference using compiled C/Cython or ONNX runtime.

### 42. What would you do if sensor noise increased?
1. Apply low-pass Butterworth filtering or an unscented Kalman filter before feature extraction.
2. Dynamically adjust warning envelopes using moving standard deviation bands ($3\sigma$).
3. Require multi-sample temporal persistence (e.g., 3 consecutive anomalous samples) before triggering a maintenance alert.

### 43. How would you reduce false alarms?
By implementing temporal persistence filtering (debouncing), requiring consensus between threshold limits and ML anomaly scores, and correlating alerts with auxiliary flight control or load switch states.

### 44. How would you validate the system?
1. Hardware-in-the-loop (HIL) testing using a programmable AC/DC power source and electronic load bank.
2. Ingesting historical electrical failure telemetry recorded during ground test rigs or airline maintenance incidents.
3. Cross-validating across diverse operational temperatures and flight phase profiles.

### 45. Could this system be deployed directly on an aircraft?
No. Direct deployment on certified aircraft requires compliance with FAA/EASA airworthiness standards, specifically DO-178C (Software Considerations in Airborne Systems) Level A/B certification, deterministic worst-case execution time guarantees, formal verification, and hardened avionics hardware qualified under DO-160G. It is suited as a ground-based condition monitoring and predictive maintenance advisory tool.
