#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("D:\\AIT614\\Project\\Air_Quality.csv")
print(df)


# In[2]:


# List of keywords
keywords = ["Asthma", "Cardiovascular", "Respiratory", "Death", "Cardiac and respiratory deaths"]

# Filter rows based on keywords in the "Name" column
filtered_data = df[df['Name'].str.contains('|'.join(keywords))]

# Display the filtered data
print(filtered_data)


# In[3]:


# Calculate the frequency of each health impact
name_frequency = filtered_data['Name'].value_counts().reset_index()
name_frequency.columns = ['Name', 'Frequency']

# Define custom colors
colors = plt.cm.Paired(np.linspace(0, 1, len(name_frequency)))

# Pie chart
plt.figure(figsize=(12, 8))
wedges, texts, autotexts = plt.pie(name_frequency['Frequency'], startangle=140, autopct='%1.1f%%', pctdistance=0.85, colors=colors, textprops=dict(color="w"))

plt.title('Health Impacts due to various pollutants')
plt.axis('equal')

# Add color boxes on the side
legend_patches = []
for i, autotext in enumerate(name_frequency['Name']):
    # Add color box
    legend_patches.append(mpatches.Patch(color=colors[i], label=autotext))

plt.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(1, 0))
plt.show()


# In[4]:


# Aggregate data by 'Time Period' and 'Name'
line_data = filtered_data.groupby(['Time Period', 'Name']).sum().reset_index()

# Extract years from 'Time Period' for sorting and plotting
line_data['Year'] = line_data['Time Period'].str.extract('(\d{4})')

# Sort the data by 'Year' and 'Name'
line_data = line_data.sort_values(by=['Year', 'Name'])

# Line plot
plt.figure(figsize=(20, 10))
for name in line_data['Name'].unique():
    plt_data = line_data[line_data['Name'] == name]
    plt.plot(plt_data['Year'], plt_data['Data Value'], marker='o', label=name)

plt.title('Trend of Health Impacts due to various pollutants over time')
plt.xlabel('Year')
plt.ylabel('Total Impact Value')
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame containing the relevant data

# List of keywords
keywords = ["Asthma", "Cardiovascular", "Respiratory", "Death", "Cardiac and respiratory deaths"]

# Filter rows based on keywords in the "Name" column
filtered_data = df[df['Name'].str.contains('|'.join(keywords))]

# Convert 'Data Value' to numeric if needed
filtered_data['Data Value'] = pd.to_numeric(filtered_data['Data Value'], errors='coerce')

# Aggregate data by 'Geo Place Name' and 'Name', calculating the average of 'Data Value'
agg_data = filtered_data.groupby(['Geo Place Name', 'Name'])['Data Value'].mean().reset_index()

# Pivot the data to create the stacked bar chart
pivot_data = agg_data.pivot(index='Geo Place Name', columns='Name', values='Data Value').reset_index()

# Stacked Bar plot
plt.figure(figsize=(12, 8))
pivot_data.plot(x='Geo Place Name', kind='bar', stacked=True, colormap='Set2', figsize=(12, 8))

plt.title('Health Impacts by Area')
plt.xlabel('Area')
plt.ylabel('Average Health Impact')
plt.xticks(rotation=90)
plt.legend(title='Health Impact', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# In[ ]:




