# %% [markdown]
# <a id="table-of-contents"></a>
# # 📖 Table of Contents
# 
# [⚙️ Setup](#setup)  
# [📐 Continuous Distributions](#continuous-distributions)  
# - [📊 Uniform](#uniform)  
# - [📈 Normal](#normal)  
# - [⏱️ Exponential](#exponential)  
# - [📦 Chi Square](#chi-square)  
# - [📘 Student t](#student-t)  
# 
# [🎲 Discrete Distributions](#discrete-distributions)  
# - [🔔 Poisson](#poisson)  
# - [⚪ Bernoulli](#bernoulli)  
# - [🎯 Binomial](#binomial)  
# 
# ___

# %% [markdown]
# <a id="setup"></a>
# # ⚙️ Setup

# %% [markdown]
# ## 📌 Glossary  
# 
# <details><summary><strong>📖 Click to Expand</strong></summary>
# 
# #### 🧠 Basic Concepts
# - 🎲 **Random Variable**: A variable whose values depend on outcomes of a random phenomenon. Usually denoted by \(X\). \(X\) can be **Discrete** or **Continuous**.  
# - 🗂️ **Probability Distribution**: Describes the probability values for each possible value a random variable can take.  
#   - For **discrete** variables: **Probability Mass Function (PMF)**  
#   - For **continuous** variables: **Probability Density Function (PDF)**  
# - 📈 **Cumulative Distribution Function (CDF)**: Describes the probability that a variable takes a value **less than or equal to** a certain value. Represents total probability from \(X = -\infty\) to \(X = x\).  
# - 📊 **Expected Value (\(\mathbb{E}[X]\))**: The long-run average value a random variable is expected to take. Also called the **Expectation** or **Mean**.  
# - 📉 **Variance (\(\text{Var}(X)\))**: Measures spread around the mean. Higher variance means more dispersion.  
# - 📏 **Standard Deviation (\(\sigma\))**: Square root of variance — shows typical deviation from the mean.  
# 
# </details>
# 

# %% [markdown]
# ## 🎲 Types of Random Variables
# 
# <details><summary><strong>📖 Click to Expand</strong></summary>
# 
# - 🔢 **Discrete Random Variable**: A variable that can take a **countable** number of values (e.g., number of customers in a store).  
#   🧪 Example: Binomial, Poisson distributions.  
# 
# - 📏 **Continuous Random Variable**: A variable that can take **infinitely many values** within a range (e.g., height, weight).  
#   🧪 Example: Normal, Exponential distributions.  
# 
# </details>
# 

# %% [markdown]
# ## 🎯 Probability
# 
# <details><summary><strong>📖 Click to Expand</strong></summary>
# 
# - 🔗 **Independence**: Two events are independent if the outcome of one **does not** affect the outcome of the other.  
# - 🎯 **Conditional Probability** (\( P(A | B) \)): The probability of event \(A\) occurring **given that** event \(B\) has already occurred.  
# - 🔄 **Bayes’ Theorem**: A formula used to update probabilities based on new information. Given prior probability \( P(A) \) and likelihood \( P(B | A) \), it calculates the posterior probability \( P(A | B) \):  
#   \[
#   P(A | B) = \frac{P(B | A) P(A)}{P(B)}
#   \]  
# - 📈 **Law of Large Numbers (LLN)**: As the number of trials increases, the sample mean **approaches** the true population mean.  
# - 🧮 **Central Limit Theorem (CLT)**: The sum or average of a **large** number of independent random variables follows a **Normal Distribution**, regardless of the original distribution.  
# 
# </details>
# 

# %% [markdown]
# ## 📐 Distribution-Specific Terms
# 
# <details><summary><strong>📖 Click to Expand</strong></summary>
# 
# - 🔀 **Skewness**: A measure of how asymmetrical a distribution is.  
#   - ➡️ **Right-skewed (positive skew)**: Long tail on the right.  
#   - ⬅️ **Left-skewed (negative skew)**: Long tail on the left.  
# - 🎢 **Kurtosis**: A measure of the "tailedness" of a distribution. High kurtosis means more extreme outliers.  
# - 🧾 **Moments**: The **nth moment** of a distribution gives information about its shape (e.g., 1st moment = mean, 2nd moment = variance).  
# 
# </details>
# 

# %% [markdown]
# ## 🧠 Estimation & Inference
# 
# <details><summary><strong>📖 Click to Expand</strong></summary>
# 
# - 📈 **Maximum Likelihood Estimation (MLE)**: A method to estimate parameters by maximizing the likelihood of observed data.  
# - 🧮 **Method of Moments (MoM)**: A parameter estimation technique that equates sample moments to theoretical moments.  
# - 📊 **Bayesian Inference**: A method of statistical inference that updates prior beliefs based on observed data.  
# - 🎯 **Confidence Interval (CI)**: A range of values that is likely to contain the true parameter value with a certain probability (e.g., 95% CI).  
# - 📉 **p-value**: The probability of observing the given data (or something more extreme) if the null hypothesis is true. A small \( p \)-value (< 0.05) suggests strong evidence **against** the null hypothesis.  
# - 🧪 **Hypothesis Testing**: A process to determine if a hypothesis is statistically significant.  
#   - ⚪ **Null Hypothesis (\( H_0 \))**: Assumes no effect or no difference.  
#   - 🔴 **Alternative Hypothesis (\( H_1 \))**: Assumes a significant effect or difference.  
# 
# </details>
# 

# %% [markdown]
# ## 💼 Applications in Data Science
# 
# <details><summary><strong>📖 Click to Expand</strong></summary>
# 
# - 🧪 **A/B Testing**: A statistical method to compare two groups and determine which performs better.  
# - 📉 **Regression Analysis**: A technique to model relationships between variables.  
# - 🎯 **Overfitting vs. Underfitting**:  
#   - 📐 **Overfitting**: The model is too complex and fits noise in the data.  
#   - 📏 **Underfitting**: The model is too simple and fails to capture patterns.  
# 
# </details>
# 

# %% [markdown]
# ___
# [Back to the top](#table-of-contents)

# %% [markdown]
# <a id="continuous-distributions"></a>
# # 📐 Continuous Distributions
# 

# %%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% [markdown]
# <a id="uniform"></a>
# ## 📊 Uniform

# %% [markdown]
# #### Sample from Distribution

# %%
import numpy as np

# Define the range [a, b]
a, b = 0, 10
size = 10000  

# Generate random samples from the Uniform distribution
samples = np.random.uniform(a, b, size)

# Print the first 10 samples as a quick check
print(samples[:10])

# %% [markdown]
# #### Key Properties

# %%
import scipy.stats as stats

# 1. Probability Density Function (PDF)
x = np.linspace(a, b, 100)
pdf_values = stats.uniform.pdf(x, a, b-a)

# 2. Cumulative Distribution Function (CDF)
cdf_values = stats.uniform.cdf(x, a, b-a)

# %%
# 3. Expected Value (Mean), Variance, and Standard Deviation
mean = np.mean(samples)
variance = np.var(samples)
std_dev = np.std(samples)

# 4. Skewness and Kurtosis
skewness = stats.skew(samples)
kurtosis = stats.kurtosis(samples)

print(f"Mean: {mean}, Variance: {variance}, Std Dev: {std_dev}")
print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")

# %%
# 5. Moments of Distribution
moment_1 = stats.moment(samples, moment=1)
moment_2 = stats.moment(samples, moment=2)
moment_3 = stats.moment(samples, moment=3)
moment_4 = stats.moment(samples, moment=4)

# Print results
print(f"Moments: 1st={moment_1}, 2nd={moment_2}, 3rd={moment_3}, 4th={moment_4}")

# %% [markdown]
# #### Visualizing Distributions

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

def plot_distribution(samples, distribution_name, dist_func=None, params=None, discrete=False):
    """
    General function to visualize a distribution.
    
    Args:
        samples (np.array): The generated random samples.
        distribution_name (str): Name of the distribution (for title/labels).
        dist_func (scipy.stats distribution, optional): Distribution function from scipy.stats (e.g., stats.norm).
        params (tuple, optional): Parameters for the distribution function (e.g., (mean, std) for normal).
        discrete (bool): Set to True for discrete distributions.
    """

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1️⃣ Histogram vs. Theoretical Distribution (PMF/PDF)
    axs[0].hist(samples, bins=50 if not discrete else np.arange(min(samples), max(samples) + 1.5) - 0.5,
                density=True, alpha=0.6, color='blue', edgecolor='black', label="Sampled Data")
    
    if dist_func and params:
        x = np.linspace(min(samples), max(samples), 100) if not discrete else np.arange(min(samples), max(samples) + 1)
        if discrete:
            y = dist_func.pmf(x, *params)
        else:
            y = dist_func.pdf(x, *params)
        axs[0].plot(x, y, 'r-', label="Theoretical")
    
    axs[0].set_title(f"Histogram vs. Theoretical {distribution_name}")
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Density / Probability")
    axs[0].legend()

    # 2️⃣ CDF Plot
    sorted_samples = np.sort(samples)
    empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    axs[1].plot(sorted_samples, empirical_cdf, marker="o", linestyle="none", label="Empirical CDF")
    
    if dist_func and params:
        theoretical_cdf = dist_func.cdf(sorted_samples, *params)
        axs[1].plot(sorted_samples, theoretical_cdf, 'r-', label="Theoretical CDF")
    
    axs[1].set_title(f"CDF of {distribution_name}")
    axs[1].set_xlabel("Value")
    axs[1].set_ylabel("Cumulative Probability")
    axs[1].legend()

    # 3️⃣ QQ-Plot (for normality check)
    stats.probplot(samples, dist="norm", plot=axs[2])
    axs[2].set_title(f"QQ-Plot for {distribution_name}")

    plt.tight_layout()
    plt.show()


# %%
plot_distribution(samples, "Uniform Distribution", stats.uniform, (0, 10))


# %% [markdown]
# #### Parameter Estimation

# %% [markdown]
# - Maximum Likelihood Estimation (MLE)

# %%
def mle_uniform(samples):
    """ MLE for Uniform Distribution: Estimates a (min) and b (max) """
    estimated_a = np.min(samples)
    estimated_b = np.max(samples)
    return estimated_a, estimated_b

# Example usage
estimated_a, estimated_b = mle_uniform(samples)
print(f"MLE Estimated a: {estimated_a}, b: {estimated_b}")


# %% [markdown]
# - Method of Moments (MoM)

# %%
def mom_uniform(samples):
    """ MoM for Uniform Distribution: Estimates a (min) and b (max) """
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)

    estimated_a = sample_mean - np.sqrt(3 * sample_var)
    estimated_b = sample_mean + np.sqrt(3 * sample_var)
    
    return estimated_a, estimated_b

# Example usage
estimated_a, estimated_b = mom_uniform(samples)
print(f"MoM Estimated a: {estimated_a}, b: {estimated_b}")


# %% [markdown]
# - Bayesian Inference

# %%
import scipy.stats as stats

def bayesian_uniform(samples, prior_range=(0, 20)):
    """ Bayesian estimation for Uniform Distribution using weakly informative priors """
    prior_a = stats.uniform(0, prior_range[1])  # Prior for a
    prior_b = stats.uniform(0, prior_range[1])  # Prior for b

    estimated_a = np.min(samples)  # Approximate MAP estimate
    estimated_b = np.max(samples)

    return estimated_a, estimated_b

# Example usage
estimated_a, estimated_b = bayesian_uniform(samples)
print(f"Bayesian Estimated a: {estimated_a}, b: {estimated_b}")


# %% [markdown]
# ___
# [Back to the top](#table-of-contents)

# %% [markdown]
# <a id="normal"></a>
# ## 📈 Normal
# 
# (Gaussian)

# %% [markdown]
# #### Sample from Distribution

# %%
import numpy as np

# Define parameters
mean = 0    
std_dev = 1  
size = 10000  

# Generate samples
samples = np.random.normal(mean, std_dev, size)

# Print first 10 samples
print(samples[:10])


# %% [markdown]
# #### Key Properties

# %%
# Define x range
x = np.linspace(min(samples), max(samples), 100)

# 1. Probability Density Function (PDF)
pdf_values = stats.norm.pdf(x, np.mean(samples), np.std(samples))

# 2. Cumulative Distribution Function (CDF)
cdf_values = stats.norm.cdf(x, np.mean(samples), np.std(samples))


# %%
# 3. Expected Value (Mean), Variance, and Standard Deviation
mean = np.mean(samples)
variance = np.var(samples)
std_dev = np.std(samples)

# 4. Skewness and Kurtosis
skewness = stats.skew(samples)
kurtosis = stats.kurtosis(samples)

print(f"Mean: {mean}, Variance: {variance}, Std Dev: {std_dev}")
print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")

# %%
# 5. Moments of Distribution
moment_1 = stats.moment(samples, moment=1)
moment_2 = stats.moment(samples, moment=2)
moment_3 = stats.moment(samples, moment=3)
moment_4 = stats.moment(samples, moment=4)

# Print results
print(f"Moments: 1st={moment_1}, 2nd={moment_2}, 3rd={moment_3}, 4th={moment_4}")

# %% [markdown]
# #### Visualizing Distributions

# %%
plot_distribution(samples, "Normal Distribution", stats.norm, (np.mean(samples), np.std(samples)))


# %% [markdown]
# #### Parameter Estimation

# %% [markdown]
# - Maximum Likelihood Estimation (MLE)

# %%
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

def mle_normal(samples):
    """ MLE for Normal Distribution """
    def neg_log_likelihood(params):
        mu, sigma = params
        return -np.sum(stats.norm.logpdf(samples, mu, sigma))
    
    init_params = [np.mean(samples), np.std(samples)]
    result = minimize(neg_log_likelihood, init_params, method="L-BFGS-B")
    return result.x  # [estimated_mu, estimated_sigma]

# Example usage
estimated_mu, estimated_sigma = mle_normal(samples)
print(f"MLE Estimated μ: {estimated_mu}, σ: {estimated_sigma}")


# %%
from scipy.optimize import minimize

def mle_normal(samples):
    """ MLE for Normal Distribution: Estimates mean and std deviation """
    def neg_log_likelihood(params):
        mu, sigma = params
        return -np.sum(stats.norm.logpdf(samples, mu, sigma))
    
    init_params = [np.mean(samples), np.std(samples)]
    result = minimize(neg_log_likelihood, init_params, method="L-BFGS-B")
    return result.x  # [estimated_mu, estimated_sigma]

# Example usage
estimated_mu, estimated_sigma = mle_normal(samples)
print(f"MLE Estimated μ: {estimated_mu}, σ: {estimated_sigma}")


# %% [markdown]
# - Method of Moments (MoM)

# %%
def mom_normal(samples):
    """ MoM for Normal Distribution: Estimates mean and std deviation """
    estimated_mu = np.mean(samples)
    estimated_sigma = np.std(samples)
    return estimated_mu, estimated_sigma

# Example usage
estimated_mu, estimated_sigma = mom_normal(samples)
print(f"MoM Estimated μ: {estimated_mu}, σ: {estimated_sigma}")


# %% [markdown]
# - Bayesian Inference

# %%
def bayesian_normal(samples, prior_mu=0, prior_sigma=10):
    """ Bayesian estimation for Normal Distribution using Normal-Gamma prior """
    n = len(samples)
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)

    posterior_mu = (prior_mu + n * sample_mean) / (n + 1)
    posterior_sigma = np.sqrt(sample_var / (n + 1))

    return posterior_mu, posterior_sigma

# Example usage
estimated_mu, estimated_sigma = bayesian_normal(samples)
print(f"Bayesian Estimated μ: {estimated_mu}, σ: {estimated_sigma}")


# %% [markdown]
# ___
# [Back to the top](#table-of-contents)

# %% [markdown]
# <a id="exponential"></a>
# ## ⏱️ Exponential
# 

# %% [markdown]
# #### Sample from Distribution

# %%
import numpy as np

# Define parameter
lambda_ = 1.0  # Rate parameter (1/mean)
size = 10000  

# Generate samples
samples = np.random.exponential(1/lambda_, size)

# Print first 10 samples
print(samples[:10])

# %% [markdown]
# #### Key Properties

# %%
# Define x range
x = np.linspace(min(samples), max(samples), 100)

# 1. Probability Density Function (PDF)
pdf_values = stats.expon.pdf(x, scale=np.mean(samples))

# 2. Cumulative Distribution Function (CDF)
cdf_values = stats.expon.cdf(x, scale=np.mean(samples))


# %%
# 3. Expected Value (Mean), Variance, and Standard Deviation
mean = np.mean(samples)
variance = np.var(samples)
std_dev = np.std(samples)

# 4. Skewness and Kurtosis
skewness = stats.skew(samples)
kurtosis = stats.kurtosis(samples)

print(f"Mean: {mean}, Variance: {variance}, Std Dev: {std_dev}")
print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")

# %%
# 5. Moments of Distribution
moment_1 = stats.moment(samples, moment=1)
moment_2 = stats.moment(samples, moment=2)
moment_3 = stats.moment(samples, moment=3)
moment_4 = stats.moment(samples, moment=4)

# Print results
print(f"Moments: 1st={moment_1}, 2nd={moment_2}, 3rd={moment_3}, 4th={moment_4}")

# %% [markdown]
# #### Visualizing Distributions

# %%
plot_distribution(samples, "Exponential Distribution", stats.expon, (0, np.mean(samples)))


# %% [markdown]
# #### Parameter Estimation

# %% [markdown]
# - Maximum Likelihood Estimation (MLE)

# %%
def mle_exponential(samples):
    """ MLE for Exponential Distribution: Estimates lambda (rate parameter) """
    estimated_lambda = 1 / np.mean(samples)
    return estimated_lambda

# Example usage
estimated_lambda = mle_exponential(samples)
print(f"MLE Estimated λ: {estimated_lambda}")


# %% [markdown]
# - Method of Moments (MoM)

# %%
def mom_exponential(samples):
    """ MoM for Exponential Distribution: Estimates lambda (rate parameter) """
    sample_mean = np.mean(samples)
    estimated_lambda = 1 / sample_mean
    return estimated_lambda

# Example usage
estimated_lambda = mom_exponential(samples)
print(f"MoM Estimated λ: {estimated_lambda}")


# %% [markdown]
# - Bayesian Inference

# %%
def bayesian_exponential(samples, alpha_prior=1, beta_prior=1):
    """ Bayesian estimation for Exponential Distribution using Gamma prior """
    n = len(samples)
    posterior_alpha = alpha_prior + n
    posterior_beta = beta_prior + np.sum(samples)

    return posterior_alpha, posterior_beta

# Example usage
posterior_alpha, posterior_beta = bayesian_exponential(samples)
print(f"Bayesian Estimated λ ~ Gamma({posterior_alpha}, {posterior_beta})")


# %% [markdown]
# ___
# [Back to the top](#table-of-contents)

# %% [markdown]
# <a id="chi-square"></a>
# ## 📦 Chi Square

# %% [markdown]
# #### Sample from Distribution

# %%
import numpy as np

# Define degrees of freedom
df = 4  
size = 10000  

# Generate samples
samples = np.random.chisquare(df, size)

# Print first 10 samples
print(samples[:10])


# %% [markdown]
# #### Key Properties

# %%
# Define x range
x = np.linspace(min(samples), max(samples), 100)

# Degrees of freedom (df) estimated from sample mean
df = np.mean(samples)

# 1. Probability Density Function (PDF)
pdf_values = stats.chi2.pdf(x, df)

# 2. Cumulative Distribution Function (CDF)
cdf_values = stats.chi2.cdf(x, df)


# %%
# 3. Expected Value (Mean), Variance, and Standard Deviation
mean = np.mean(samples)
variance = np.var(samples)
std_dev = np.std(samples)

# 4. Skewness and Kurtosis
skewness = stats.skew(samples)
kurtosis = stats.kurtosis(samples)

print(f"Mean: {mean}, Variance: {variance}, Std Dev: {std_dev}")
print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")

# %%
# 5. Moments of Distribution
moment_1 = stats.moment(samples, moment=1)
moment_2 = stats.moment(samples, moment=2)
moment_3 = stats.moment(samples, moment=3)
moment_4 = stats.moment(samples, moment=4)

# Print results
print(f"Moments: 1st={moment_1}, 2nd={moment_2}, 3rd={moment_3}, 4th={moment_4}")

# %% [markdown]
# #### Visualizing Distributions

# %%
plot_distribution(samples, "Chi-Square Distribution", stats.chi2, (np.mean(samples),))


# %% [markdown]
# #### Parameter Estimation

# %% [markdown]
# - Maximum Likelihood Estimation (MLE)

# %%
def mle_chi_square(samples):
    """ MLE for Chi-Square Distribution: Estimates degrees of freedom (df) """
    estimated_df = np.mean(samples)
    return estimated_df

# Example usage
estimated_df = mle_chi_square(samples)
print(f"MLE Estimated df: {estimated_df}")


# %% [markdown]
# - Method of Moments (MoM)

# %%
def mom_chi_square(samples):
    """ MoM for Chi-Square Distribution: Estimates degrees of freedom (df) """
    estimated_df = np.mean(samples)
    return estimated_df

# Example usage
estimated_df = mom_chi_square(samples)
print(f"MoM Estimated df: {estimated_df}")


# %% [markdown]
# - Bayesian Inference

# %%
def bayesian_chi_square(samples, alpha_prior=1, beta_prior=1):
    """ Bayesian estimation for Chi-Square Distribution using Gamma prior """
    n = len(samples)
    posterior_alpha = alpha_prior + n / 2
    posterior_beta = beta_prior + np.sum(samples) / 2

    return posterior_alpha, posterior_beta

# Example usage
posterior_alpha, posterior_beta = bayesian_chi_square(samples)
print(f"Bayesian Estimated df ~ Gamma({posterior_alpha}, {posterior_beta})")


# %% [markdown]
# ___
# [Back to the top](#table-of-contents)

# %% [markdown]
# <a id="student-t"></a>
# ## 📘 Student t

# %% [markdown]
# #### Sample from Distribution

# %%
import numpy as np

# Define degrees of freedom
df = 10  
size = 10000  

# Generate samples
samples = np.random.standard_t(df, size)

# Print first 10 samples
print(samples[:10])


# %% [markdown]
# #### Key Properties

# %%
# Define x range
x = np.linspace(min(samples), max(samples), 100)

# Degrees of freedom (df) estimated from sample variance
df = len(samples) - 1

# 1. Probability Density Function (PDF)
pdf_values = stats.t.pdf(x, df)

# 2. Cumulative Distribution Function (CDF)
cdf_values = stats.t.cdf(x, df)


# %%
# 3. Expected Value (Mean), Variance, and Standard Deviation
mean = np.mean(samples)
variance = np.var(samples)
std_dev = np.std(samples)

# 4. Skewness and Kurtosis
skewness = stats.skew(samples)
kurtosis = stats.kurtosis(samples)

print(f"Mean: {mean}, Variance: {variance}, Std Dev: {std_dev}")
print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")

# %%
# 5. Moments of Distribution
moment_1 = stats.moment(samples, moment=1)
moment_2 = stats.moment(samples, moment=2)
moment_3 = stats.moment(samples, moment=3)
moment_4 = stats.moment(samples, moment=4)

# Print results
print(f"Moments: 1st={moment_1}, 2nd={moment_2}, 3rd={moment_3}, 4th={moment_4}")

# %% [markdown]
# #### Visualizing Distributions

# %%
plot_distribution(samples, "Student's t-Distribution", stats.t, (len(samples) - 1,))


# %% [markdown]
# #### Parameter Estimation

# %% [markdown]
# - Maximum Likelihood Estimation (MLE)

# %%
def mle_student_t(samples):
    """ MLE for Student's t-Distribution: Estimates degrees of freedom (df) """
    def neg_log_likelihood(df):
        return -np.sum(stats.t.logpdf(samples, df))
    
    result = minimize(neg_log_likelihood, x0=[10], method="L-BFGS-B", bounds=[(1, None)])
    return result.x[0]

# Example usage
estimated_df = mle_student_t(samples)
print(f"MLE Estimated df: {estimated_df}")


# %% [markdown]
# - Method of Moments (MoM)

# %%
def mom_student_t(samples):
    """ MoM for Student's t-Distribution: Estimates degrees of freedom (df) """
    sample_var = np.var(samples)
    
    def solve_df(df):
        return df / (df - 2) - sample_var  # Equation to solve

    from scipy.optimize import fsolve
    estimated_df = fsolve(solve_df, x0=10)[0]
    return estimated_df

# Example usage
estimated_df = mom_student_t(samples)
print(f"MoM Estimated df: {estimated_df}")


# %% [markdown]
# - Bayesian Inference

# %%
def bayesian_student_t(samples, alpha_prior=1, beta_prior=1):
    """ Bayesian estimation for Student's t-Distribution using Gamma prior """
    n = len(samples)
    posterior_alpha = alpha_prior + n / 2
    posterior_beta = beta_prior + np.sum(samples**2) / 2

    return posterior_alpha, posterior_beta

# Example usage
posterior_alpha, posterior_beta = bayesian_student_t(samples)
print(f"Bayesian Estimated df ~ Gamma({posterior_alpha}, {posterior_beta})")


# %% [markdown]
# ___
# [Back to the top](#table-of-contents)

# %% [markdown]
# <a id="discrete-distributions"></a>
# # 🎲 Discrete Distributions
# 

# %% [markdown]
# <a id="poisson"></a>
# ## 🔔 Poisson
# 

# %% [markdown]
# #### Sample from Distribution

# %%
import numpy as np

# Define parameter
lambda_ = 3  # Average number of events per interval
size = 10000  

# Generate samples
samples = np.random.poisson(lambda_, size)

# Print first 10 samples
print(samples[:10])


# %% [markdown]
# #### Key Properties

# %%
# Define x range (only integer values for discrete distribution)
x = np.arange(min(samples), max(samples)+1)

# Lambda estimated as sample mean
lambda_ = np.mean(samples)

# 1. Probability Mass Function (PMF)
pmf_values = stats.poisson.pmf(x, lambda_)

# 2. Cumulative Distribution Function (CDF)
cdf_values = stats.poisson.cdf(x, lambda_)


# %%
# 3. Expected Value (Mean), Variance, and Standard Deviation
mean = np.mean(samples)
variance = np.var(samples)
std_dev = np.std(samples)

# 4. Skewness and Kurtosis
skewness = stats.skew(samples)
kurtosis = stats.kurtosis(samples)

print(f"Mean: {mean}, Variance: {variance}, Std Dev: {std_dev}")
print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")

# %%
# 5. Moments of Distribution
moment_1 = stats.moment(samples, moment=1)
moment_2 = stats.moment(samples, moment=2)
moment_3 = stats.moment(samples, moment=3)
moment_4 = stats.moment(samples, moment=4)

# Print results
print(f"Moments: 1st={moment_1}, 2nd={moment_2}, 3rd={moment_3}, 4th={moment_4}")

# %% [markdown]
# #### Visualizing Distributions

# %%
plot_distribution(samples, "Poisson Distribution", stats.poisson, (np.mean(samples),), discrete=True)


# %% [markdown]
# #### Parameter Estimation

# %% [markdown]
# - Maximum Likelihood Estimation (MLE)

# %%
def mle_poisson(samples):
    """ MLE for Poisson Distribution: Estimates lambda (mean rate of occurrences) """
    estimated_lambda = np.mean(samples)
    return estimated_lambda

# Example usage
estimated_lambda = mle_poisson(samples)
print(f"MLE Estimated λ: {estimated_lambda}")


# %% [markdown]
# - Method of Moments (MoM)

# %%
def mom_poisson(samples):
    """ MoM for Poisson Distribution: Estimates lambda (mean rate of occurrences) """
    estimated_lambda = np.mean(samples)
    return estimated_lambda

# Example usage
estimated_lambda = mom_poisson(samples)
print(f"MoM Estimated λ: {estimated_lambda}")


# %% [markdown]
# - Bayesian Inference

# %%
def bayesian_poisson(samples, alpha_prior=1, beta_prior=1):
    """ Bayesian estimation for Poisson Distribution using Gamma prior """
    posterior_alpha = alpha_prior + np.sum(samples)
    posterior_beta = beta_prior + len(samples)

    return posterior_alpha, posterior_beta

# Example usage
posterior_alpha, posterior_beta = bayesian_poisson(samples)
print(f"Bayesian Estimated λ ~ Gamma({posterior_alpha}, {posterior_beta})")


# %% [markdown]
# ___
# [Back to the top](#table-of-contents)

# %% [markdown]
# <a id="bernoulli"></a>
# ## ⚪ Bernoulli
# 

# %%
import numpy as np

# Define probability of success
p = 0.5  
size = 10000  

# Generate samples (0 or 1 outcomes)
samples = np.random.binomial(1, p, size)

# Print first 10 samples
print(samples[:10])


# %% [markdown]
# #### Sample from Distribution

# %% [markdown]
# #### Key Properties

# %%
# Define x values (0 and 1 only for Bernoulli)
x = np.array([0, 1])

# Probability of success (p) estimated from sample mean
p = np.mean(samples)

# 1. Probability Mass Function (PMF)
pmf_values = stats.bernoulli.pmf(x, p)

# 2. Cumulative Distribution Function (CDF)
cdf_values = stats.bernoulli.cdf(x, p)


# %%
# 3. Expected Value (Mean), Variance, and Standard Deviation
mean = np.mean(samples)
variance = np.var(samples)
std_dev = np.std(samples)

# 4. Skewness and Kurtosis
skewness = stats.skew(samples)
kurtosis = stats.kurtosis(samples)

print(f"Mean: {mean}, Variance: {variance}, Std Dev: {std_dev}")
print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")

# %%
# 5. Moments of Distribution
moment_1 = stats.moment(samples, moment=1)
moment_2 = stats.moment(samples, moment=2)
moment_3 = stats.moment(samples, moment=3)
moment_4 = stats.moment(samples, moment=4)

# Print results
print(f"Moments: 1st={moment_1}, 2nd={moment_2}, 3rd={moment_3}, 4th={moment_4}")

# %% [markdown]
# #### Visualizing Distributions

# %%
plot_distribution(samples, "Bernoulli Distribution", stats.bernoulli, (np.mean(samples),), discrete=True)


# %% [markdown]
# #### Parameter Estimation

# %% [markdown]
# - Maximum Likelihood Estimation (MLE)

# %%
def mle_bernoulli(samples):
    """ MLE for Bernoulli Distribution: Estimates probability of success (p) """
    estimated_p = np.mean(samples)
    return estimated_p

# Example usage
estimated_p = mle_bernoulli(samples)
print(f"MLE Estimated p: {estimated_p}")


# %% [markdown]
# - Method of Moments (MoM)

# %%
def mom_bernoulli(samples):
    """ MoM for Bernoulli Distribution: Estimates probability of success (p) """
    estimated_p = np.mean(samples)
    return estimated_p

# Example usage
estimated_p = mom_bernoulli(samples)
print(f"MoM Estimated p: {estimated_p}")


# %% [markdown]
# - Bayesian Inference

# %%
def bayesian_bernoulli(samples, alpha_prior=1, beta_prior=1):
    """ Bayesian estimation for Bernoulli Distribution using Beta prior """
    successes = np.sum(samples)
    failures = len(samples) - successes

    posterior_alpha = alpha_prior + successes
    posterior_beta = beta_prior + failures

    return posterior_alpha, posterior_beta

# Example usage
posterior_alpha, posterior_beta = bayesian_bernoulli(samples)
print(f"Bayesian Estimated p ~ Beta({posterior_alpha}, {posterior_beta})")


# %% [markdown]
# ___
# [Back to the top](#table-of-contents)

# %% [markdown]
# <a id="binomial"></a>
# ## 🎯 Binomial
# 

# %% [markdown]
# #### Sample from Distribution

# %%
import numpy as np

# Define parameters
n = 10   # Number of trials
p = 0.5  # Probability of success
size = 10000  

# Generate samples
samples = np.random.binomial(n, p, size)

# Print first 10 samples
print(samples[:10])

# %% [markdown]
# #### Key Properties

# %%
# Define x range (integer values from 0 to max observed trials)
x = np.arange(min(samples), max(samples)+1)

# Number of trials (n) estimated as max observed value
n = max(samples)

# Probability of success (p) estimated from sample mean divided by n
p = np.mean(samples) / n

# 1. Probability Mass Function (PMF)
pmf_values = stats.binom.pmf(x, n, p)

# 2. Cumulative Distribution Function (CDF)
cdf_values = stats.binom.cdf(x, n, p)


# %%
# 3. Expected Value (Mean), Variance, and Standard Deviation
mean = np.mean(samples)
variance = np.var(samples)
std_dev = np.std(samples)

# 4. Skewness and Kurtosis
skewness = stats.skew(samples)
kurtosis = stats.kurtosis(samples)

print(f"Mean: {mean}, Variance: {variance}, Std Dev: {std_dev}")
print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")

# %%
# 5. Moments of Distribution
moment_1 = stats.moment(samples, moment=1)
moment_2 = stats.moment(samples, moment=2)
moment_3 = stats.moment(samples, moment=3)
moment_4 = stats.moment(samples, moment=4)

# Print results
print(f"Moments: 1st={moment_1}, 2nd={moment_2}, 3rd={moment_3}, 4th={moment_4}")

# %% [markdown]
# #### Visualizing Distributions

# %%
plot_distribution(samples, "Binomial Distribution", stats.binom, (max(samples), np.mean(samples) / max(samples)), discrete=True)


# %% [markdown]
# #### Parameter Estimation

# %% [markdown]
# - Maximum Likelihood Estimation (MLE)

# %%
def mle_binomial(samples, n):
    """ MLE for Binomial Distribution: Estimates probability of success (p) given n trials """
    estimated_p = np.mean(samples) / n
    return estimated_p

# Example usage
n = max(samples)  # Assuming n is the max observed value
estimated_p = mle_binomial(samples, n)
print(f"MLE Estimated p: {estimated_p} (given n={n})")


# %% [markdown]
# - Method of Moments (MoM)

# %%
def mom_binomial(samples, n):
    """ MoM for Binomial Distribution: Estimates probability of success (p) given n trials """
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)

    estimated_p = sample_mean / n
    return estimated_p

# Example usage
n = max(samples)  # Assuming n is the max observed value
estimated_p = mom_binomial(samples, n)
print(f"MoM Estimated p: {estimated_p} (given n={n})")

# %% [markdown]
# - Bayesian Inference

# %%
from scipy.stats import beta

def bayesian_bernoulli(samples, alpha_prior=1, beta_prior=1):
    """ Bayesian estimation for Bernoulli distribution """
    successes = np.sum(samples)  # Number of 1s (successes)
    failures = len(samples) - successes  # Number of 0s (failures)
    
    posterior_alpha = alpha_prior + successes
    posterior_beta = beta_prior + failures
    
    return posterior_alpha, posterior_beta

# Example usage
posterior_alpha, posterior_beta = bayesian_bernoulli(samples)
print(f"Posterior Beta(α, β): α={posterior_alpha}, β={posterior_beta}")


# %%
def bayesian_binomial(samples, n, alpha_prior=1, beta_prior=1):
    """ Bayesian estimation for Binomial Distribution using Beta prior """
    successes = np.sum(samples)

    posterior_alpha = alpha_prior + successes
    posterior_beta = beta_prior + (n * len(samples) - successes)

    return posterior_alpha, posterior_beta

# Example usage
n = max(samples)  # Assuming n is the max observed value
posterior_alpha, posterior_beta = bayesian_binomial(samples, n)
print(f"Bayesian Estimated p ~ Beta({posterior_alpha}, {posterior_beta})")


# %% [markdown]
# ___
# [Back to the top](#table-of-contents)


