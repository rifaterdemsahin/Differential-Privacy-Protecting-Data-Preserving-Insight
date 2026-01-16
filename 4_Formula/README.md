# 📚 4_Formula: Guides & Best Practices

- **Premise:** Don't reinvent the wheel.
- **Content:** Essential guides, formulas, and code snippets.
- **Conclusion:** Solves challenges efficiently and ensures high quality.

## 🧪 The Formulas (Algorithms)

### Differential Privacy Definition
A randomized mechanism M satisfies **ε-differential privacy** if for all datasets D1, D2 differing in one record, and all outputs S:
```
Pr[M(D1) ∈ S] ≤ e^ε × Pr[M(D2) ∈ S]
```

### The Laplace Mechanism
To release a statistic f(D) with ε-differential privacy:
```python
def laplace_mechanism(true_value, sensitivity, epsilon):
    """
    Add Laplace noise to achieve differential privacy.
    
    Args:
        true_value: The actual computed value
        sensitivity: Maximum change in value from adding/removing one record
        epsilon: Privacy budget parameter
    
    Returns:
        Noisy value with ε-DP guarantee
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale)
    return true_value + noise
```

**Example**: Private average age
```python
# Sensitivity of average: (max_age - min_age) / n
sensitivity = 100 / len(dataset)  # assuming ages 0-100
epsilon = 1.0
true_avg = np.mean(ages)
private_avg = laplace_mechanism(true_avg, sensitivity, epsilon)
```

### DP-SGD: Differentially Private Stochastic Gradient Descent

#### Standard SGD
```python
# Standard SGD step
gradients = compute_gradients(batch, model)
optimizer.apply_gradients(zip(gradients, model.variables))
```

#### DP-SGD with Gradient Clipping and Noise
```python
# DP-SGD step
def dp_sgd_step(batch, model, optimizer, noise_multiplier, l2_norm_clip):
    # Compute per-sample gradients (not averaged)
    per_sample_gradients = compute_per_sample_gradients(batch, model)
    
    # Clip each gradient to maximum L2 norm
    clipped_gradients = []
    for grad in per_sample_gradients:
        grad_norm = tf.norm(grad)
        clip_factor = tf.minimum(1.0, l2_norm_clip / grad_norm)
        clipped = grad * clip_factor
        clipped_gradients.append(clipped)
    
    # Average clipped gradients
    avg_clipped = tf.reduce_mean(clipped_gradients, axis=0)
    
    # Add Gaussian noise (Laplace can also be used)
    noise_stddev = noise_multiplier * l2_norm_clip
    noisy_gradients = []
    for grad in avg_clipped:
        noise = tf.random.normal(tf.shape(grad), mean=0, stddev=noise_stddev)
        noisy_gradients.append(grad + noise)
    
    # Apply noisy gradients
    optimizer.apply_gradients(zip(noisy_gradients, model.variables))
```

### Privacy Budget Accounting

#### Composition Theorem
If we run k mechanisms, each with ε-DP, total privacy degrades to kε-DP (basic composition).

**Advanced composition** (tighter bound):
```
Total privacy ≤ ε√(2k log(1/δ)) + kε(e^ε - 1)
```

#### Privacy Budget Calculation
```python
def calculate_privacy_budget(steps, noise_multiplier, batch_size, dataset_size, delta=1e-5):
    """
    Calculate epsilon using Rényi Differential Privacy accounting.
    
    Args:
        steps: Number of training steps
        noise_multiplier: Noise scale parameter
        batch_size: Training batch size
        dataset_size: Total training samples
        delta: Failure probability
    
    Returns:
        epsilon: Privacy budget spent
    """
    # Sampling probability
    q = batch_size / dataset_size
    
    # Use RDP accountant (simplified approximation)
    # Real implementation would use TensorFlow Privacy's RDP accountant
    orders = [1 + x / 10. for x in range(1, 100)]
    rdp = compute_rdp(q, noise_multiplier, steps, orders)
    epsilon = get_privacy_spent(orders, rdp, delta)[0]
    
    return epsilon
```

### Key Parameters

#### Noise Multiplier
-   **Relationship to epsilon**: Higher noise → Lower epsilon → Stronger privacy
-   **Typical values**: 0.5 to 2.0
-   **Formula**: `noise_stddev = noise_multiplier × l2_norm_clip`

#### Gradient Clipping Norm
-   **Purpose**: Limits influence of any single example
-   **Typical values**: 0.5 to 1.5
-   **Effect**: Bounds sensitivity of gradient computation

#### Delta (δ)
-   **Definition**: Failure probability of privacy guarantee
-   **Typical value**: 1/n² where n is dataset size (e.g., 1e-5 for 10,000 samples)
-   **Interpretation**: Privacy holds except with probability δ

## 📐 Privacy-Utility Tradeoff Equation

Accuracy loss ≈ O(√(log(1/δ)) / (εn))

Where:
-   n = dataset size
-   ε = privacy budget
-   δ = failure probability

**Key insight**: Larger datasets allow better privacy-utility tradeoffs!
