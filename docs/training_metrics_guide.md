# DIAYN Training Metrics and Optimization

## Core Metrics

### 1. Policy Loss (`train/policy_loss`)
- **Measures**: How well the agent maximizes expected reward
- **Ideal**: Decreases over time with some oscillation
- **Issues**: 
  - Increasing: Unstable training
  - Zero: Numerical instability

### 2. Discriminator Loss (`train/discriminator_loss`)
- **Measures**: Skill prediction accuracy
- **Ideal**: Stable or slowly increasing
- **Issues**:
  - Near 0: Skills too easy to distinguish
  - High: Skills too similar

## Key Hyperparameters

### Learning Rate (`lr`)
- **Default**: 1e-4
- **Effect**: Controls weight updates
- **Too High**: Unstable training
- **Too Low**: Slow learning

### Batch Size (`batch_size`)
- **Default**: 256
- **Effect**: Number of samples per update
- **Larger**: More stable, but slower
- **Smaller**: Faster, noisier updates

### Replay Buffer Size (`replay_size`)
- **Default**: 10,000
- **Effect**: Experience diversity
- **Larger**: More stable, but slower
- **Smaller**: Faster, less diverse

## Optimization Tips

1. **Start Small**
   - Use smaller environments first
   - Fewer skills initially (4-8)
   - Shorter episodes

2. **Monitor**
   - Watch for NaN/Inf values
   - Check gradient norms
   - Track entropy

3. **Adjust**
   - Increase batch size if unstable
   - Reduce learning rate if oscillating
   - Scale intrinsic rewards if needed

## Common Issues

1. **NaN/Inf Values**
   - Reduce learning rate
   - Add gradient clipping
   - Check for numerical stability

2. **No Learning**
   - Increase batch size
   - Check reward scaling
   - Verify network architecture

3. **Overfitting**
   - Add dropout
   - Increase buffer size
   - Add L2 regularization
