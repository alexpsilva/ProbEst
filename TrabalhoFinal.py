import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optimize
import numpy as np
import pandas as pd
import csv
import math

precision = 3

data = []
with open('E:\Downloads\Dados-medicos.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        # Get rid of extra spaces
        filtered_row = []
        for item in row:
            if item:
                try:
                    filtered_row.append(float(item))
                except:
                    continue
        
        if filtered_row:
            data.append(filtered_row)

labels = ['idade', 'peso', 'carga_final', 'vo2_max']

def remove_outliers(input_data):
    df = pd.DataFrame.from_records(input_data, columns=labels)
    
    high_quantile = df.quantile(0.75)
    low_quantile = df.quantile(0.25)
    iqr = high_quantile - low_quantile
    
    outlier_filter = (df >= (low_quantile - 1.5*iqr)) & (df <= (high_quantile + 1.5*iqr))
    return df[outlier_filter].dropna().reset_index(drop=True)

data = remove_outliers(data).values.tolist()

# (2.1) Histograma e Função Distribuição Empírica

def generate_histogram(input_data, ax, index=0, title=None):
    column_data = [i[index] for i in input_data]
    
    # Compute the bins to be used
    standard_deviation = np.std(column_data, ddof=1)
    bin_size = 3.49*standard_deviation/np.cbrt(len(column_data))
    bins = range(int(min(column_data)), int(max(column_data)), int(bin_size))
    
    # Plot histogram
    ax.hist(column_data, bins=bins, density=True)
    if title:
        ax.set_title(title)

def generate_histogram(input_data, ax, index=0, title=None):
    column_data = [i[index] for i in input_data]
    
    # Compute the bins to be used
    standard_deviation = np.std(column_data, ddof=1)
    bin_size = 3.49*standard_deviation/np.cbrt(len(column_data))
    bins = range(int(min(column_data)), int(max(column_data)), int(bin_size))
    
    # Plot histogram
    ax.hist(column_data, bins=bins, density=True)
    if title:
        ax.set_title(title)

def generate_ecdf(input_data, ax, index=0, title=None):
    # Compute the Empiric Cumulative Distribution Function (ECDF)
    sorted_data = sorted([i[index] for i in input_data])
    ecdf_values = {}
    for value_index, value in enumerate(sorted_data):
        if value in ecdf_values:
            ecdf_values[value] += 1
        else:
            ecdf_values[value] = value_index+1
    
    ecdf = [float(ecdf_values[i])/len(sorted_data) for i in sorted_data]
    
    # Plot ECDF
    ax.plot(sorted_data, ecdf)
    if title:
        ax.set_title(title)

fig, axs = plt.subplots(1, 4, figsize=(15, 5))

generate_ecdf(data, axs[0], index=0, title=u'Idade (anos)')
generate_ecdf(data, axs[1], index=1, title=u'Peso (kg)')
generate_ecdf(data, axs[2], index=2, title=u'Carga Final (Watt)')
generate_ecdf(data, axs[3], index=3, title=u'VO2 Máximo (mL/kg/min)')

plt.show()

# (2.2) Média, Variância e BoxPlot

def generate_boxplot(input_data, ax, index=0, title=None):
    # Compute mean and variance
    column_data = [i[index] for i in input_data]
    mean = 0.0
    for i in column_data:
        mean += i
    mean = round(mean/len(column_data), precision)
    
    variance = round(np.var(column_data), precision)
    
    # Plot boxplot
    ax.boxplot(column_data)
    if title:
        ax.set_title(u'%s\n\nMédia: %s\nVariância: %s' % (title, mean, variance))

fig, axs = plt.subplots(1, 4, figsize=(15, 5))

generate_boxplot(data, axs[0], index=0, title=u'Idade (anos)')
generate_boxplot(data, axs[1], index=1, title=u'Peso (kg)')
generate_boxplot(data, axs[2], index=2, title=u'Carga Final (Watt)')
generate_boxplot(data, axs[3], index=3, title=u'VO2 Máximo (mL/kg/min)')

plt.show()

# (2.3) Parametrizando distribuições

fitted_params_by_column = {}

def calculate_log_likelihood(column_df, distribution, loc, scale, shape=None):
    # Compute the log likelihood of a given 'distribution', with a given set of 'loc', 'scale' and (optinaly) 'shape' parameters 
    likelihood = 0.0
    for index, i in column_df.iteritems():
        if shape is None:
            value = np.log(distribution(i, loc, scale))
        else:
            value = np.log(distribution(i, shape, loc, scale))
        
        likelihood += value
    return likelihood

def generate_distribution(ax, distribution, loc, scale, shape=None, interval=[1, 100], step=5, factor=1.0, title='Distribution'):
    # Plot the given 'distribution', with parameters 'loc', 'scale' and (optinaly) 'shape'
    x_data = range(int(interval[0]), int(interval[1]), int(step))
    x_data = [float(i)*factor for i in x_data]
    
    if shape is None:
        y_data = [distribution(i, loc, scale) for i in x_data]
    else:
        y_data = [distribution(i, shape, loc, scale) for i in x_data]
    
    ax.plot(x_data, y_data)
    
    if title:
        new_title = u'%s\n\nLoc: %s\nScale: %s' %(title, round(loc, precision), round(scale, precision))
        if shape:
            new_title = u'%s\nShape: %s' %(new_title, round(shape, precision))
        ax.set_title(new_title)

def generate_fitted_distribution(input_data, ax, index, distribution, title=None):
    df = pd.DataFrame.from_records(input_data, columns=labels)
    column_df = df[labels[index]]
    
    # Fit to distribution
    fitted_distribution = distribution.fit(column_df)
    
    result = {}
    if len(fitted_distribution) > 2:
        result['shape'] = fitted_distribution[0]
        result['loc'] = fitted_distribution[1]
        result['scale'] = fitted_distribution[2]
    else:
        result['loc'] = fitted_distribution[0]
        result['scale'] = fitted_distribution[1]
    
    fitted_params_by_column.setdefault(index, {}).setdefault(distribution, {})
    fitted_params_by_column[index][distribution] = result
    
    # Plot distribution and hitogram
    generate_histogram(input_data, ax, index=index)
    
    min_value = column_df.min()
    max_value = column_df.max()
    if title:
        log_likelihood = calculate_log_likelihood(column_df, distribution.pdf, **result)
        new_title = u'%s\n\nLogLikelihood: %s' %(title, round(log_likelihood, precision))
    else:
        new_title = title
    generate_distribution(ax, distribution.pdf, interval=[min_value, max_value], step=1, title=new_title, **result)

fig, axs = plt.subplots(1, 4, figsize=(15, 5))

generate_fitted_distribution(data, axs[0], index=0, distribution=stats.expon, title=u'Exponencial')
generate_fitted_distribution(data, axs[1], index=0, distribution=stats.norm, title=u'Gaussiana')
generate_fitted_distribution(data, axs[2], index=0, distribution=stats.lognorm, title=u'Lognormal')
generate_fitted_distribution(data, axs[3], index=0, distribution=stats.weibull_min, title=u'Weibull')

plt.show()

fig, axs = plt.subplots(1, 4, figsize=(15, 5))

generate_fitted_distribution(data, axs[0], index=1, distribution=stats.expon, title=u'Exponencial')
generate_fitted_distribution(data, axs[1], index=1, distribution=stats.norm, title=u'Gaussiana')
generate_fitted_distribution(data, axs[2], index=1, distribution=stats.lognorm, title=u'Lognormal')
generate_fitted_distribution(data, axs[3], index=1, distribution=stats.weibull_min, title=u'Weibull')

plt.show()

fig, axs = plt.subplots(1, 4, figsize=(15, 5))

generate_fitted_distribution(data, axs[0], index=2, distribution=stats.expon, title=u'Exponencial')
generate_fitted_distribution(data, axs[1], index=2, distribution=stats.norm, title=u'Gaussiana')
generate_fitted_distribution(data, axs[2], index=2, distribution=stats.lognorm, title=u'Lognormal')
generate_fitted_distribution(data, axs[3], index=2, distribution=stats.weibull_min, title=u'Weibull')

plt.show()

fig, axs = plt.subplots(1, 4, figsize=(15, 5))

generate_fitted_distribution(data, axs[0], index=3, distribution=stats.expon, title=u'Exponencial')
generate_fitted_distribution(data, axs[1], index=3, distribution=stats.norm, title=u'Gaussiana')
generate_fitted_distribution(data, axs[2], index=3, distribution=stats.lognorm, title=u'Lognormal')
generate_fitted_distribution(data, axs[3], index=3, distribution=stats.weibull_min, title=u'Weibull')

plt.show()

# (2.4) Gráfico QQ-Plot ou Probability-Plot

def generate_qqplot(input_data, ax, index, distribution, title=None):
    column_data = [i[index] for i in input_data]
    distribution_params = fitted_params_by_column[index][distribution]
    if len(distribution_params.keys()) > 2:
        params = (distribution_params['shape'], distribution_params['loc'], distribution_params['scale'])
    else:
        params = (distribution_params['loc'], distribution_params['scale'])
    
    stats.probplot(column_data, dist=distribution, sparams=params, plot=ax)
    
    if title:
        ax.set_title(title)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

generate_qqplot(data, axs[0], index=0, distribution=stats.expon, title=u'Exponencial')
generate_qqplot(data, axs[1], index=0, distribution=stats.norm, title=u'Gaussiana')
generate_qqplot(data, axs[2], index=0, distribution=stats.lognorm, title=u'Lognormal')
generate_qqplot(data, axs[3], index=0, distribution=stats.weibull_min, title=u'Weibull')

plt.show()

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

generate_qqplot(data, axs[0], index=1, distribution=stats.expon, title=u'Exponencial')
generate_qqplot(data, axs[1], index=1, distribution=stats.norm, title=u'Gaussiana')
generate_qqplot(data, axs[2], index=1, distribution=stats.lognorm, title=u'Lognormal')
generate_qqplot(data, axs[3], index=1, distribution=stats.weibull_min, title=u'Weibull')

plt.show()

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

generate_qqplot(data, axs[0], index=2, distribution=stats.expon, title=u'Exponencial')
generate_qqplot(data, axs[1], index=2, distribution=stats.norm, title=u'Gaussiana')
generate_qqplot(data, axs[2], index=2, distribution=stats.lognorm, title=u'Lognormal')
generate_qqplot(data, axs[3], index=2, distribution=stats.weibull_min, title=u'Weibull')

plt.show()

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

generate_qqplot(data, axs[0], index=3, distribution=stats.expon, title=u'Exponencial')
generate_qqplot(data, axs[1], index=3, distribution=stats.norm, title=u'Gaussiana')
generate_qqplot(data, axs[2], index=3, distribution=stats.lognorm, title=u'Lognormal')
generate_qqplot(data, axs[3], index=3, distribution=stats.weibull_min, title=u'Weibull')

plt.show()

# (2.5) Teste de Hipótese

distribution_to_name = {
    stats.expon: u'Exponencial', 
    stats.norm: u'Gaussiana', 
    stats.lognorm: u'Lognormal', 
    stats.weibull_min: u'Weibull'
}

def make_ks_test(input_data, max_index, distributions):
    results = {}
    for index in range(max_index+1):
        index_results = {}
        for distribution in distributions:
            column_data = [i[index] for i in input_data]
            distribution_params = fitted_params_by_column[index][distribution]
            if len(distribution_params.keys()) > 2:
                params = (distribution_params['shape'], distribution_params['loc'], distribution_params['scale'])
            else:
                params = (distribution_params['loc'], distribution_params['scale'])

            distribution_name = distribution_to_name[distribution]
            index_results[distribution_name] = stats.kstest(column_data, distribution.cdf, params)
        results[index] = index_results
    return results

ks_results = make_ks_test(data, 3, [stats.expon, stats.norm, stats.lognorm, stats.weibull_min])
for index in range(4):
    print(u'%s:' % labels[index])
    for distribution, distribution_name in distribution_to_name.iteritems():
        value = ks_results[index][distribution_name]
        print(u'  %s:  Teste=%s  |  P-Valor=%s' % (distribution_name, round(value[0], precision), round(value[1], precision) ))
    print(u'\n')

# (2.6) Análise de dependência entre as variáveis, modelo de regressão

def depencecy_to_vo2(input_data, ax, index, y_label):
    vo2_column_data = [i[3] for i in input_data]
    y_column_data = [i[index] for i in input_data]
    
    # Plot linear regression's result
    linear_regression = stats.linregress(vo2_column_data, y_column_data)
    max_vo2 = max(vo2_column_data)
    slope = linear_regression[0]
    intercept = linear_regression[1]
    ax.plot([0.0, max_vo2], [intercept, slope*max_vo2 + intercept], color='red')
    
    # Plot Scater plot
    ax.scatter(vo2_column_data, y_column_data)
    ax.set_xlabel(u'VO2')
    ax.set_ylabel(y_label)
    
    # Print Pearson's coeficient
    ax.set_title(u'Coeficiente de Pearson: %s' % round(linear_regression[2], precision))

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

depencecy_to_vo2(data, axs[0], index=0, y_label=u'Idade')
depencecy_to_vo2(data, axs[1], index=1, y_label=u'Peso')
depencecy_to_vo2(data, axs[2], index=2, y_label=u'Carga Máxima')

plt.show()

# (2.7) Inferência Bayesiana

def validate_hypotesis(value, hypotesis):
    valid_occurence = True
    
    hypotesis_min = hypotesis.get('min')
    if hypotesis_min is not None and value <= float(hypotesis_min):
        valid_occurence = False

    hypotesis_max = hypotesis.get('max')
    if hypotesis_max is not None and value > float(hypotesis_max):
        valid_occurence = False
    
    return valid_occurence

def calculate_prior(input_data, index, hypotesis):
    lower_than_equal_ocurrences = []    
    for value in input_data:
        if validate_hypotesis(value[index], hypotesis):
            lower_than_equal_ocurrences.append(value)
    
    return float(len(lower_than_equal_ocurrences))/len(input_data), lower_than_equal_ocurrences

def calculate_likelihood(input_data, index, vo2_hypotesis, hypotesis):
    prior, filtered_data = calculate_prior(input_data, index, hypotesis)
    
    lower_than_equal_ocurrences = 0
    for value in filtered_data:
        if validate_hypotesis(value[3], hypotesis=vo2_hypotesis):
            lower_than_equal_ocurrences += 1        
    
    return float(lower_than_equal_ocurrences)/len(filtered_data)

def calculate_posterior(input_data, index, vo2_hypotesis, hypotesis_list, preset_priors=None):
    table = []
    bayes_numerators = []
    for i, hypotesis in enumerate(hypotesis_list):
        if preset_priors is None:
            prior, filtered_data = calculate_prior(input_data, index, hypotesis)
        else:
            prior = preset_priors[i]
        likelihood = calculate_likelihood(input_data, index, vo2_hypotesis, hypotesis)
        bayes_numerator = prior*likelihood
        
        bayes_numerators.append(bayes_numerator)
        table.append({'hypotesis': hypotesis, 'prior': prior, 'likelihood': likelihood, 'bayes_numerator': bayes_numerator})
    
    numerators_sum = sum(bayes_numerators)
    for row in table:
        row['posterior'] = row['bayes_numerator']/numerators_sum
    
    return table

def print_table(table):
    
    print(u'   hipótese        |   prior  | likelihood | numerator |  posterior')
    print(u'-------------------------------------------------------------------')
    
    row_label_size = 0
    for row in table:
        new_precision = precision + 2
        format_string = '{:.%sf}' % new_precision
        prior = format_string.format(row['prior'])
        likelihood = format_string.format(row['likelihood'])
        bayes_numerator = format_string.format(row['bayes_numerator'])
        posterior =format_string.format(row['posterior'])
        
        full_str = u'%s  |  %s   |  %s  |  %s' % (prior, likelihood, bayes_numerator, posterior)
        
        hypotesis = row['hypotesis']
        
        row_label = u''
        if hypotesis.get('min') is None:
            row_label = u'%s       ' % row_label
        elif len(str(hypotesis['min'])) == 2:
            row_label = u'%s%s    <' % (row_label, hypotesis['min'])
        elif len(str(hypotesis['min'])) == 3:
            row_label = u'%s%s   <' % (row_label, hypotesis['min'])
        elif len(str(hypotesis['min'])) == 5:
            row_label = u'%s%s <' % (row_label, hypotesis['min'])
        
        row_label = u'%s X' % (row_label)
        
        if hypotesis.get('max') is None:
            row_label = u'%s         ' % row_label
        elif len(str(hypotesis['max'])) == 2:
            row_label = u'%s   <= %s ' % (row_label, hypotesis['max'])
        elif len(str(hypotesis['max'])) == 3:
            row_label = u'%s   <= %s' % (row_label, hypotesis['max'])
        elif len(str(hypotesis['max'])) == 5:
            row_label = u'%s <= %s' % (row_label, hypotesis['max'])
        
        full_str = u'%s | %s' % (row_label, full_str)
        print(full_str)

hypotesis = [
    {'max': 50},
    {'min': 50, 'max': 100},
    {'min': 100, 'max': 150},
    {'min': 150, 'max': 200},
    {'min': 200, 'max': 250},
    {'min': 250, 'max': 300},
    {'min': 300}
]

vo2_under_35 = calculate_posterior(data, 2, {'max': 35}, hypotesis)

print(u'                          VO2 < 35                             ')
print_table(vo2_under_35)
print('\n')

print(u'                          VO2 >= 35                            ')
print_table(calculate_posterior(data, 2, {'min': 35}, hypotesis))
print('\n')

print(u'                    VO2 >= 35 | VO2 < 35                       ')
updated_prior = [i['posterior'] for i in vo2_under_35]
print_table(calculate_posterior(data, 2, {'min': 35}, hypotesis, updated_prior))

    