# %% [markdown]
# <a id="table-of-contents"></a>
# # 📖 Table of Contents
# 
# [📊 Law of Large Numbers](#law-of-large-numbers)  
# [📈 Central Limit Theorem](#central-limit-theorem)  
# <hr style="border: none; height: 1px; background-color: #ddd;" />

# %% [markdown]
# <a id="law-of-large-numbers"></a>
# 
# # 📊 Law of Large Numbers
# 

# %% [markdown]
# <details><summary><strong>📖 Click to Expand</strong></summary>
# 
# <p>The <strong>Law of Large Numbers (LLN)</strong>: As the size of a sample increases, the sample mean (average) will converge to the expected value (population mean).</p>
# 
# <h5>📌 Key Points</h5>
# <ul>
#   <li>The larger the sample size, the closer the sample mean gets to the true population mean.</li>
#   <li>LLN applies to repeated independent experiments, like rolling dice or flipping coins.</li>
# </ul>
# 
# <h5>🎲 Examples</h5>
# <ol>
#   <li>In a coin toss, as the number of tosses increases, the proportion of heads converges to 0.5.</li>
#   <li>When rolling a die, the average result converges to the expected value of 3.5 as the number of rolls increases.</li>
#   <li>In estimating the average age in a population, taking larger samples gives a more accurate result.</li>
# </ol>
# 
# </details>
# 

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Number of trials
num_trials = 10000

# Simulate dice rolls
dice_rolls = np.random.randint(1, 7, size=num_trials)

# Compute cumulative averages
cumulative_averages = np.cumsum(dice_rolls) / np.arange(1, num_trials + 1)

# Plot the convergence
plt.figure(figsize=(10, 6))
plt.plot(cumulative_averages, label="Cumulative Average")
plt.axhline(y=3.5, color='r', linestyle='--', label="Expected Value (3.5)")
plt.title("Law of Large Numbers - Dice Rolls")
plt.xlabel("Number of Rolls")
plt.ylabel("Cumulative Average")
plt.legend()
plt.grid()
plt.show()

# %%
# Generate a population
population_mean = 50
population_std = 10
population = np.random.normal(loc=population_mean, scale=population_std, size=100000)

# Draw increasing sample sizes and compute means
sample_sizes = np.arange(1, 10001)
sample_means = [np.mean(np.random.choice(population, size=n)) for n in sample_sizes]

# Plot the convergence
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, sample_means, label="Sample Mean")
plt.axhline(y=population_mean, color='r', linestyle='--', label="Population Mean (50)")
plt.title("Law of Large Numbers - Population Sampling")
plt.xlabel("Sample Size")
plt.ylabel("Sample Mean")
plt.legend()
plt.grid()
plt.show()


# %%
# Number of rolls
num_rolls = 10000

# Simulate die rolls
rolls = np.random.randint(1, 7, size=num_rolls)

# Calculate cumulative averages
cumulative_averages = np.cumsum(rolls) / np.arange(1, num_rolls + 1)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(cumulative_averages, label="Cumulative Average", color="blue")
plt.axhline(y=3.5, color='red', linestyle='--', label="Expected Value (3.5)")
plt.title("Law of Large Numbers - Rolling a Die")
plt.xlabel("Number of Rolls")
plt.ylabel("Cumulative Average")
plt.legend()
plt.grid()
plt.show()


# %% [markdown]
# #### `Observations`
# 1. In both simulations, as the sample size increases:
#    - The sample mean converges to the true population mean (or expected value).
# 2. The convergence is faster for distributions with less variability.
# 3. The LLN demonstrates that larger samples are more reliable for estimating population parameters.
# 

# %% [markdown]
# ___
# [Back to the top](#table-of-contents)

# %% [markdown]
# <a id="central-limit-theorem"></a>
# 
# # 📈 Central Limit Theorem

# %% [markdown]
# <details><summary><strong>📖 Click to Expand</strong></summary>
# 
# <p>The <strong>Central Limit Theorem (CLT)</strong> is a fundamental concept in statistics that explains the behavior of sample means when drawing random samples from a population. The Central Limit Theorem states that:</p>
# 
# <ol>
#   <li>When you take random samples of size \(n\) from a population with any distribution (e.g., skewed, uniform, etc.), the distribution of the sample means will tend to be approximately normal (bell-shaped) as the sample size becomes large.</li>
#   <li>This result holds regardless of the shape of the population distribution.</li>
# </ol>
# 
# <h5>📌 Key Components</h5>
# <ul>
#   <li><strong>Population Mean (μ):</strong> The average value in the population.</li>
#   <li><strong>Population Standard Deviation (σ):</strong> The spread or variability in the population.</li>
#   <li><strong>Sample Mean (x̄):</strong> The average value in a random sample.</li>
#   <li><strong>Sampling Distribution of the Sample Mean:</strong> The distribution of sample means from all possible samples of size \( n \).</li>
# </ul>
# 
# <h5>❓ Why is the CLT Important?</h5>
# <ol>
#   <li>It allows us to use the properties of the normal distribution even if the population distribution is unknown or non-normal.</li>
#   <li>It provides the foundation for many statistical methods, including hypothesis testing and confidence intervals.</li>
#   <li>It ensures that sample means are predictable and centered around the population mean.</li>
# </ol>
# 
# <h5>✅ Conditions for the CLT</h5>
# <ol>
#   <li>The samples must be <strong>randomly selected</strong>.</li>
#   <li>The sample size (\(n\)) should be sufficiently large. As a rule of thumb, \(n > 30\) is often sufficient, but heavily skewed distributions may require larger samples.</li>
#   <li>The samples should be <strong>independent</strong> of each other.</li>
# </ol>
# 
# <h5>📐 Formula for Standard Error of the Mean</h5>
# 
# <p>The standard deviation of the sampling distribution of the sample mean, known as the <strong>Standard Error (SE)</strong>, is given by:</p>
# 
# $$
# SE = \frac{\sigma}{\sqrt{n}}
# $$
# 
# <p>Where:</p>
# <ul>
#   <li>\(\sigma\): The standard deviation of the population.</li>
#   <li>\(n\): The sample size.</li>
# </ul>
# 
# </details>
# 

# %%
# Parameters
population_mean = 35  # The mean (average) value of the simulated population.
population_std = 15 # The standard deviation of the population, indicating how spread out.
sample_size = 50 # The number of individuals (data points) in each random sample drawn from the population.
num_samples = 10000 # The total number of samples to draw from the population.
num_bins = 30 # The number of bins (intervals) used to create histograms for visualizing the distributions.

# %%
# Distributions to plot
distributions = {
    "Exponential": lambda: np.random.exponential(scale=population_std, size=100000) + population_mean,
    "Uniform": lambda: np.random.uniform(low=population_mean - 3*population_std, 
                                         high=population_mean + 3*population_std, size=100000),
    "Normal": lambda: np.random.normal(loc=population_mean, scale=population_std, size=100000),
    "Lognormal": lambda: np.random.lognormal(mean=np.log(population_mean), sigma=0.5, size=100000),
    "Poisson": lambda: np.random.poisson(lam=population_mean, size=100000),
    "Chi-Square": lambda: np.random.chisquare(df=10, size=100000),
    "Beta": lambda: np.random.beta(a=2, b=5, size=100000) * 70  # Scale beta for a better range
}

# %%
# Initialize the figure
fig, axes = plt.subplots(len(distributions), 2, figsize=(14, 4 * len(distributions)))

# Iterate through distributions
for i, (dist_name, dist_func) in enumerate(distributions.items()):
    # Generate population data
    population = dist_func()
    
    # Collect sample means
    sample_means = [np.mean(np.random.choice(population, sample_size)) for _ in range(num_samples)]
    
    # Plot the population distribution
    axes[i, 0].hist(population, bins=num_bins, color='skyblue', edgecolor='black')
    axes[i, 0].set_title(f'{dist_name} Distribution (Population)')
    axes[i, 0].set_xlabel('Value')
    axes[i, 0].set_ylabel('Frequency')
    
    # Plot the sample means distribution
    axes[i, 1].hist(sample_means, bins=num_bins, color='lightgreen', edgecolor='black')
    axes[i, 1].set_title(f'{dist_name} Distribution (Sample Means)')
    axes[i, 1].set_xlabel('Sample Mean')
    axes[i, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# %% [markdown]
# ___
# [Back to the top](#table-of-contents)


