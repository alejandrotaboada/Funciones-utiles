from scipy.stats import chi2
from scipy.stats import chi2_contingency

def prueba_chi_2(df, feature, target,prob = 0.95):
    cross = pd.crosstab(df[feature], df[target])
    vals = {}
    
    #tabla de contingencia
    stat, p, dof, expected = chi2_contingency(cross)
    critical = chi2.ppf(prob, dof)
    vals['feature'] = feature
    vals['target'] = target
    vals['prob'] = prob
    vals['critical'] = critical
    vals['stat'] = stat
    vals['significance'] = 1.0 - prob
    vals['diff'] = abs(stat) - critical
    
    if abs(stat) >= critical:
        vals['Dependientes'] = True
    else:
        vals['Dependientes'] = False
    
    return vals