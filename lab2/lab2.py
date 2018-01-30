import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np

#reeading vechicle.csv
data=pd.read_csv("vehicles.csv")
data=data.dropna()

#extracting current fleet colunmn
currentFleet=pd.DataFrame(data['Current fleet'])
currentFleet.dropna()

#extracting new fleet colun
newFleet=pd.DataFrame(data['New Fleet'])
newFleet.dropna()
cols=[i for i in range(0,len(currentFleet))]

#inserting ID column
currentFleet.insert(loc=0,column='ID',value=cols)
newFleet.insert(loc=0,column='ID',value=cols)

sns.set(color_codes=True)

'''plotting scatter_plot
currentFleet_scatter=sns.lmplot(currentFleet.columns[0],currentFleet.columns[1],data=currentFleet,fit_reg=False)
currentFleet_scatter.savefig("currentFleet_scatter.pdf",bbox_inches='tight')
plt.clf()
newFleet_scatter=sns.lmplot(newFleet.columns[0],newFleet.columns[1],data=newFleet,fit_reg=False)
newFleet_scatter.savefig('newFleet_scatter.pdf',bbox_inches='tight')

#plotting histogram
plt.clf()
currentFleet_histogram=sns.distplot(currentFleet.T[1],bins=7,kde=False,rug=True).get_figure()
axes=plt.gca()
axes.set_xlabel("Current Fleet")
axes.set_ylabel('Probability')

currentFleet_histogram.savefig('currentFleet_histogram.pdf',bbox_inches='tight')
newFleet_histogram=sns.distplot(newFleet.T[1],bins=7,kde=False,rug=True).get_figure()
axes.set_xlabel("New Fleet")
axes.set_ylabel('probability')
newFleet_histogram.savefig('new_fleet_histogram.pdf',bbox_inches='tight')
'''


#calculating standard deviation
currentFleet_std=np.std(currentFleet.values.T[1])

newFleet_std=np.std(newFleet.values.T[1])
print('currentFleet std: %.3f, newFleet std: %.3f'%(currentFleet_std,newFleet_std))

#bootstrapping to find std low and upper bound,
def bootstrap(data,iteration):
	data=data.values.T[1]
	samples=np.random.choice(data,replace=True,size=[iteration,len(data)])
	
	data_std=np.std(data)
	stds=[]
	for s in samples:
		stds.append(np.std(s))
	
	stds=np.array(stds)
	upper,lower=np.percentile(stds,[2.95,97.5])
	return (lower,data_std,upper)


cols=['Lower_bound', 'mean','Upper_bound']
currentFleet_bootstrap=[]
newFleet_bootstrap=[]

for r,i in enumerate(range(100,10000,1000)):
	current_boot=bootstrap(currentFleet,i)
	currentFleet_bootstrap.append([i,current_boot[0],"lower"])
	currentFleet_bootstrap.append([i,current_boot[1],"mean"])
	currentFleet_bootstrap.append([i,current_boot[2],'upper'])
	
	new_boot=bootstrap(newFleet,i)
	newFleet_bootstrap.append([i,new_boot[0],"lower"])
	newFleet_bootstrap.append([i,new_boot[1],"mean"])
	newFleet_bootstrap.append([i,new_boot[2],"upper"])
	


currentFleet_bootstrap=pd.DataFrame(currentFleet_bootstrap,columns=["iterations",'mean','value'])
newFleet_bootstrap=pd.DataFrame(newFleet_bootstrap,columns=['iterations','mean','value'])
current_boot=sns.lmplot(currentFleet_bootstrap.columns[0],currentFleet_bootstrap.columns[1],data=currentFleet_bootstrap,hue='value')
current_boot.savefig("currentFleet_bootstrap.pdf",bbox_inches='tight')

newFleet_boot=sns.lmplot(newFleet_bootstrap.columns[0],newFleet_bootstrap.columns[1],data=newFleet_bootstrap,hue='value')
newFleet_boot.savefig("newFleet_bootstrap.pdf",bbox_inches='tight')

plt.show()



