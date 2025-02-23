# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import statsmodels.api as sm

# load dataset
data=pd.read_csv("Countries.csv")
# to read first five rows of data
data.head()
# information of data
data.info()
# Stastical analysis of data
data.describe()
# Data Cleaning
# removing the duplicate rows in a data
print("Total rows in a data:",len(data))

data=data.drop_duplicates()
print("After removing the duplicates:",len(data))
# Checking null values and filling the null values by median and mode
data.isnull().sum()
# Missing values Bar Plot
#Displays the percentage of missing values for each column, helping identify data quality issues.
# Calculate missing values percentage
miss_value_per = pd.DataFrame(data.isnull().sum() / len(data) * 100, columns=["Missing Percentage"])
miss_value_per.reset_index(inplace=True)  # Reset index for proper labeling
miss_value_per.rename(columns={"index": "Columns"}, inplace=True)  

fig = px.bar(
    miss_value_per,
    x="Columns", 
    y="Missing Percentage",
    color="Missing Percentage",  
    color_continuous_scale="viridis",  
)

fig.update_layout(
    xaxis_title="Columns",
    yaxis_title="Percentage",
    title="Missing Values in Percentage",
    coloraxis_colorbar_title="Missing %",  
    showlegend=True  
)

fig.show()
# filling the null values
def replace_null(data, cols):
    for col in cols:
        if data[col].dtype in ['int64', 'float64']:  
            median = data[col].median()
            data[col] = data[col].fillna(median)  
        else:
            mode = data[col].mode()[0]
            data[col] = data[col].fillna(mode)  

cols = ['Agriculture (% GDP)', 'Ease of Doing Business', 'Education Expenditure (% GDP)',
        'Export (% GDP)', 'GDP', 'Health Expenditure (% GDP)', 'Import (% GDP)',
        'Industry (% GDP)', 'Inflation Rate', 'R&D', 'Service (% GDP)', 'Unemployment',
        'Export', 'Import', 'Education Expenditure', 'Health Expenditure', 'Net Trade',
        'GDP Per Capita']

replace_null(data, cols)

# Check if there are still missing values
print(data.isnull().sum())
# GDP Contribution Breakdown over the Years
# Shows how agriculture, industry, and services contribute to GDP over time.

data_grouped = data.groupby('Year')[['Agriculture (% GDP)', 'Industry (% GDP)', 'Service (% GDP)']].mean()

fig = px.bar(data, x='Year', 
             y=['Agriculture (% GDP)', 'Industry (% GDP)', 'Service (% GDP)'], 
             title="GDP Composition Over the Years",
             labels={"value": "Percentage of GDP", "Year": "Year"},
             barmode='stack')

fig.show()
#Ease of Doing Business vs. GDP
#Illustrates the relationship between ease of doing business scores and GDP, with a trendline for better insight.
X = data["Ease of Doing Business"]
y = data["GDP"]
X = sm.add_constant(X)  # Add constant for intercept
model = sm.OLS(y, X).fit()
data["Trendline"] = model.predict(X)

fig = px.scatter(data, x="Ease of Doing Business", y="GDP",
                 title="Ease of Doing Business vs GDP",
                 labels={"Ease of Doing Business": "Ease of Doing Business Score",
                         "GDP": "Gross Domestic Product"},
                 color="Continent Name", 
                 hover_name="Country Name",
                 opacity=0.7)

# Add the trendline manually with a distinct color
fig.add_scatter(x=data["Ease of Doing Business"], y=data["Trendline"],
                mode='lines', name="OLS Trendline", 
                line=dict(color='red', width=2, dash='dash'))

# Show figure
fig.show()
# Net Trade Balance by Continent
# Highlights the distribution and variability of net trade across different continents.
fig = px.box(data, x='Continent Name', y='Net Trade', 
             title="Net Trade Balance Across Continents",
             labels={"Net Trade": "Net Trade Balance ($)", "Continent Name": "Continent"},color='Continent Name')

fig.show()
# Health vs. Education Expenditure
#Shows how countries allocate GDP to health and education, with bubble sizes representing GDP.
fig = px.scatter(data, x='Education Expenditure (% GDP)', y='Health Expenditure (% GDP)', 
                 size='GDP', color='Continent Name', hover_name='Country Name',
                 title="Health vs Education Expenditure",
                 labels={"Education Expenditure (% GDP)": "Education Expenditure (% GDP)",
                         "Health Expenditure (% GDP)": "Health Expenditure (% GDP)"})

fig.show()
# GDP Over the Years
# Tracks GDP growth trends of the highest GDP nations over time.
top_15_countries = data[data['Year'] == data['Year'].max()].nlargest(15, 'GDP')['Country Name']

data_top_15 = data[data['Country Name'].isin(top_15_countries)]

# Create the line plot
fig = px.line(data_top_15, x='Year', y='GDP', color='Country Name', 
              title="GDP Trend Over the Years (Top 15 Countries)",
              labels={"GDP": "Gross Domestic Product", "Year": "Year"})
fig.show()
#Population Growth over the Years
# Illustrates population growth trends using a stepwise approach for clarity.
top_10_countries = data.groupby('Country Name')['Population'].sum().nlargest(10).index
data_top_10 = data[data['Country Name'].isin(top_10_countries)]

fig = px.line(data_top_10, x='Year', y='Population', color='Country Name', 
              title="Population Growth Over the Years (Top 10 Countries)",
              line_shape='vh',  # Vertical and horizontal steps
              labels={"Population": "Population", "Year": "Year"})
fig.show()
# Population Density vs GDP Per Capita
# Reveals the strength and direction of the relationship between these two variables.
corr_matrix = data[['Population Density', 'GDP Per Capita']].corr().round(2)
fig = ff.create_annotated_heatmap(
    z=corr_matrix.values, 
    x=list(corr_matrix.columns), 
    y=list(corr_matrix.index),
    colorscale='RdBu',
    showscale=True)

fig.update_layout(title="Correlation between Population Density and GDP Per Capita")
fig.show()
# Average Unemployment by Continent
# Compares unemployment levels across different continents.
fig = px.bar(data.groupby('Continent Name')['Unemployment'].mean().reset_index(), 
             x='Continent Name', y='Unemployment', 
             title="Average Unemployment by Continent", 
             labels={"Unemployment": "Average Unemployment", "Continent Name": "Continent"},
            color="Continent Name")
fig.show()
# GDP Distribution by Continent
# Shows the proportion of global GDP contributed by each continent.
fig = px.pie(data.groupby('Continent Name')['GDP'].sum().reset_index(), 
             names='Continent Name', values='GDP', 
             title="GDP Distribution by Continent")
fig.show()
# Distribution of Inflation Rate
# Displays the frequency of different inflation rate levels among countries.
fig = px.histogram(data, x='Inflation Rate', nbins=20, 
                   title="Distribution of Inflation Rates",
                   labels={"Inflation Rate": "Inflation Rate (%)"})
fig.show()
# Compute statistical moments for GDP
# List of numeric columns
numeric_columns = [
    'Agriculture (% GDP)', 'Ease of Doing Business', 'Education Expenditure (% GDP)',
    'Export (% GDP)', 'GDP', 'Health Expenditure (% GDP)', 'Import (% GDP)',
    'Industry (% GDP)', 'Inflation Rate', 'R&D', 'Service (% GDP)', 'Unemployment',
    'Population', 'Land', 'Export', 'Import', 'Education Expenditure',
    'Health Expenditure', 'Net Trade', 'GDP Per Capita', 'Population Density'
]

# Compute statistical moments for each column
for col in numeric_columns:
    mean = data[col].mean()
    variance = data[col].var()
    skewness = data[col].skew()
    kurtosis = data[col].kurtosis()
    
    print(f"{col} - Mean: {mean:.2f}, Variance: {variance:.2f}, Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")

# Violin Plot - GDP Distribution Across Continents
# Combines a box plot and KDE to show GDP spread and density per continent.
# Violin plot for GDP distribution per continent
fig = px.violin(data, x="Continent Name", y="GDP", box=True, points="all",
                title="GDP Distribution Across Continents",
                labels={"GDP": "Gross Domestic Product ($)", "Continent Name": "Continent"},
                color="Continent Name")
fig.show()
# Pair Plot - Key Economic Indicators
# Highlights relationships between GDP, GDP per capita, inflation rate, unemployment, and population.
features = ['GDP', 'GDP Per Capita', 'Population', 'Inflation Rate', 'Unemployment']
sns.pairplot(data[features])
plt.suptitle("Pair Plot of Key Economic Indicators", y=1.02)
plt.show()
