# ✅ 7_Testing: Validation & Quality Assurance

- **Premise:** A project is only complete when proven to work.
- **Content:** Testing scripts and documentation.
- **Conclusion:** Guarantees quality and confirms objectives are met.

## 🧪 Validation Strategy

### Test Metrics

#### 1. Baseline Accuracy (No Privacy)
**Definition**: Model accuracy with standard training.
```python
baseline_acc = standard_model.evaluate(X_test, y_test)
```
**Baseline**: Should match expected performance for architecture (~98% for MNIST)

#### 2. Private Model Accuracy
**Definition**: Model accuracy with DP-SGD training.
```python
private_acc = dp_model.evaluate(X_test, y_test)
```
**Target**: >85% of baseline (e.g., >83% if baseline is 98%)

#### 3. Privacy Budget (Epsilon)
**Definition**: Total privacy budget spent during training.
```python
epsilon = compute_privacy_budget(steps, noise_multiplier, batch_size, 
                                 dataset_size, delta)
```
**Target**: ε < 10.0 (preferably ε < 1.0 for strong privacy)

#### 4. Accuracy Gap
**Definition**: Difference between baseline and private accuracy.
```python
accuracy_gap = baseline_acc - private_acc
```
**Target**: <10% (acceptable tradeoff)

#### 5. Privacy-Utility Ratio
**Definition**: Accuracy retained per unit of privacy spent.
```python
privacy_utility_ratio = private_acc / epsilon
```
**Target**: Maximize this ratio (higher is better)

#### 6. Membership Inference Attack Success Rate
**Definition**: Probability attacker correctly identifies if example was in training set.
```python
attack_success_rate = membership_inference_attack(model, train_data, test_data)
```
**Baseline (no privacy)**: ~70-80% success rate  
**Target (with privacy)**: ~50-55% (near random guessing)

### Comprehensive Evaluation Table

The notebook generates a detailed comparison:

| Epsilon | Noise Multiplier | Clipping Norm | Accuracy | Accuracy Gap | Privacy Level |
|---------|------------------|---------------|----------|--------------|---------------|
| ∞ (None) | 0.0 | - | 98.2% | 0.0% | No Privacy |
| 10.0 | 0.5 | 1.0 | 96.8% | 1.4% | Weak Privacy |
| 5.0 | 0.8 | 1.0 | 95.2% | 3.0% | Moderate Privacy |
| 1.0 | 1.2 | 1.0 | 91.5% | 6.7% | Strong Privacy |
| 0.5 | 1.5 | 1.0 | 87.3% | 10.9% | Very Strong Privacy |

### Privacy Guarantee Verification

#### Test 1: Distinguishability Test
Verify that adding/removing one record doesn't significantly change output:
```python
def verify_privacy_guarantee(mechanism, dataset, epsilon):
    # Run mechanism on original dataset
    output1 = mechanism(dataset)
    
    # Remove one random record
    modified_dataset = remove_random_record(dataset)
    output2 = mechanism(modified_dataset)
    
    # Check if outputs are similar (within e^epsilon factor)
    ratio = probability_ratio(output1, output2)
    
    assert ratio <= np.exp(epsilon), f"Privacy violated: ratio={ratio:.2f}"
```

#### Test 2: Laplace Noise Correctness
Verify Laplace mechanism adds correct amount of noise:
```python
def test_laplace_mechanism():
    true_value = 50.0
    sensitivity = 1.0
    epsilon = 1.0
    
    samples = [laplace_mechanism(true_value, sensitivity, epsilon) 
               for _ in range(10000)]
    
    # Expected scale: sensitivity / epsilon = 1.0
    measured_scale = np.std(samples - true_value) / np.sqrt(2)
    expected_scale = sensitivity / epsilon
    
    assert abs(measured_scale - expected_scale) < 0.1
```

#### Test 3: Gradient Clipping
Verify all gradients are properly clipped:
```python
def test_gradient_clipping(gradients, clip_norm):
    for grad in gradients:
        grad_norm = tf.norm(grad).numpy()
        assert grad_norm <= clip_norm + 1e-6, f"Gradient not clipped: {grad_norm}"
```

### Membership Inference Attack Test

Evaluate privacy by testing if attacker can identify training examples:
```python
def membership_inference_attack(model, train_data, test_data):
    """
    Test if model leaks training membership information.
    
    Returns:
        Attack accuracy (50% = random guessing = perfect privacy)
    """
    # Compute loss on training and test examples
    train_losses = compute_losses(model, train_data)
    test_losses = compute_losses(model, test_data)
    
    # Threshold-based attack: classify as "in training" if loss < threshold
    threshold = np.median(train_losses + test_losses)
    
    train_predictions = (train_losses < threshold).astype(int)  # Should be 1
    test_predictions = (test_losses < threshold).astype(int)    # Should be 0
    
    # Calculate attack accuracy
    correct_train = np.mean(train_predictions == 1)
    correct_test = np.mean(test_predictions == 0)
    attack_accuracy = (correct_train + correct_test) / 2
    
    return attack_accuracy * 100
```

**Interpretation**:
-   **No privacy**: Attack accuracy ~70-80%
-   **Strong privacy**: Attack accuracy ~50-55% (near random)

### Privacy-Utility Tradeoff Curve

Generate curve showing accuracy vs epsilon:
```python
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
accuracies = []
attack_success = []

for eps in epsilons:
    if eps == float('inf'):
        # No privacy baseline
        model = train_standard_model()
    else:
        model = train_dp_model(epsilon=eps)
    
    acc = evaluate(model, test_data)
    accuracies.append(acc)
    
    attack_rate = membership_inference_attack(model, train_data, test_data)
    attack_success.append(attack_rate)

# Plot
plt.plot(epsilons[:-1], accuracies[:-1], label='Accuracy')
plt.plot(epsilons[:-1], attack_success[:-1], label='Attack Success Rate')
plt.xlabel('Privacy Budget (ε)')
plt.ylabel('Percentage (%)')
plt.legend()
```

### Validation Cells in Notebook
The notebook contains specific cells to:
1.  Train baseline model (no privacy)
2.  Train DP models with different epsilon values
3.  Calculate privacy budget for each configuration
4.  Measure accuracy gap across privacy levels
5.  Perform membership inference attacks
6.  Verify Laplace mechanism correctness
7.  Test gradient clipping implementation
8.  Generate privacy-utility tradeoff curves
9.  Create comprehensive comparison tables
10. Visualize noise impact on model outputs
