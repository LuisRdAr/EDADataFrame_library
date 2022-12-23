import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

class EDA_DataFrame():
    def __init__(self, df:pd.DataFrame):
    
        self.dataframe = df

        cols = self.dataframe.columns.to_list()

        dtypes = []
        notnulls = []
        notnulls_p = []
        unique = []
        unique_p = []
        for col in cols:
            dtypes.append(str(self.dataframe[col].dtype))
            notnulls.append(self.dataframe[col].notnull().sum())
            notnulls_p.append(
                round(self.dataframe[col].notnull().sum()/self.dataframe.shape[0], 3))
            unique.append(self.dataframe[col].nunique())
            unique_p.append(
                round(self.dataframe[col].nunique()/self.dataframe.shape[0], 3))  

        df = pd.DataFrame({"Column": cols, "Dtype": dtypes, "Non_Null": notnulls, 
                            "Non_Null_perc": notnulls_p, "Unique": unique, 
                            "Unique_perc": unique_p})
        
        print("DataFrame's shape: {} columns by {} rows" \
            .format(self.dataframe.shape[1], self.dataframe.shape[0]))
        print(round(df,2), end = "\n\n")
        print("Number of incomplete rows (with one or more null values): {} out of {} ({}%)" \
            .format(self.dataframe[self.dataframe.isna().any(axis=1)].shape[0],
            self.dataframe.shape[0], 
            100*round(self.dataframe[self.dataframe.isna().any(axis=1)].shape[0]/self.dataframe.shape[0], 4)))

        num_cols = df[(df["Dtype"].str.contains("float")) | \
            (df["Dtype"].str.contains("int"))].loc[:,"Column"].to_list()

        cat_cols = df[((df["Dtype"] == "object") | (df["Dtype"] == "bool")) 
                            & (df["Unique_perc"] < 0.05)].loc[:,"Column"].to_list()

        self.cat_columns = cat_cols
        self.num_columns = num_cols


    def search_duplicates(self):
        print("This DataFrame has {} duplicated rows considering all its attributes and," \
            .format(self.dataframe[self.dataframe.duplicated(keep = False)].shape[0]))
        for i, col1 in enumerate(self.dataframe.columns):
            n_duplicated = self.dataframe[self.dataframe \
                .loc[:, self.dataframe.columns != col1].duplicated(keep = False)].shape[0]
            print("\t {} duplicated rows when excluding only {},".format(n_duplicated, col1))
            for j, col2 in enumerate(self.dataframe.columns):
                n_duplicated2 = self.dataframe[self.dataframe. \
                loc[:, (self.dataframe.columns != col1) & (self.dataframe.columns != col2)] \
                    .duplicated(keep = False)].shape[0]
                if n_duplicated2 != 0:
                    print("\t\t but {} duplicated rows when excluding the {}-{} pair," \
                        .format(n_duplicated2, col1, col2))


    def univariate_analysis(self):

        cat_plots = len(self.cat_columns)
        if cat_plots > 0:
            rows_plots = math.ceil(cat_plots/3)
            fig, ax = plt.subplots(rows_plots, 3, figsize = (15, rows_plots*5), sharey = True)
            fig.suptitle('Univariate Analysis of the Categorical Attributes', fontsize=16)
            for i, col in enumerate(self.cat_columns):
                x = self.dataframe[col].value_counts().index.to_list()
                y = self.dataframe[col].value_counts(normalize = True).values
                sns.barplot(x = x, y = y, ax = ax[math.floor(i/3)][int(i%3)]).set_title(col)
                ax[math.floor(i/3)][int(i%3)].set_ylabel("Percentage") 
                
                for bar in ax[math.floor(i/3)][int(i%3)].patches:
                    ax[math.floor(i/3)][int(i%3)].annotate(format(bar.get_height(), '.3f'),
                            (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va='center',
                            size=10, xytext=(0, 5),
                            textcoords='offset points');
            plt.show()

        num_plots = len(self.num_columns)
        if num_plots > 0:
            rows_plots = math.ceil(num_plots/3)
            fig2, ax2 = plt.subplots(rows_plots, 3, figsize = (18, rows_plots*5))
            fig2.suptitle('Univariate Analysis of the Numerical Attributes', fontsize=16)
            for i, col in enumerate(self.num_columns):
                sns.kdeplot(data = self.dataframe[self.num_columns], x = col, ax = ax2[math.floor(i/3)][int(i%3)])
                ax[math.floor(i/3)][int(i%3)].set_ylabel("Percentage") 
                
                for bar in ax[math.floor(i/3)][int(i%3)].patches:
                    ax[math.floor(i/3)][int(i%3)].annotate(format(bar.get_height(), '.3f'),
                            (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va='center',
                            size=10, xytext=(0, 5),
                            textcoords='offset points');
            plt.show()

        num_plots = len(self.num_columns)
        if num_plots > 0:
            rows_plots = math.ceil(num_plots/3)
            fig3, ax3 = plt.subplots(rows_plots, 3, figsize = (15, rows_plots*5))
            fig3.suptitle('Visualization of Outliers in Numerical Attributes', fontsize=16)
            for i, col in enumerate(self.num_columns):
                sns.boxplot(data = self.dataframe[self.num_columns], x = col, ax = ax3[math.floor(i/3)][int(i%3)], fliersize = 0.5)
            plt.show()
    
    
    def correlation_matrix(self):

        sns.heatmap(self.dataframe[self.num_columns].corr(), annot = True)
    

    def outliers(self, attr = "All") -> pd.DataFrame:

        if attr == "All":
            searched_attrs = self.num_columns
        elif (type(attr) == str):
            if (attr.upper() in [col.upper() for col in self.num_columns]):
                i = [num_col.upper() for num_col in self.num_columns].index(attr.upper())
                searched_attrs = [self.num_columns[i]]
            else:
                print("The given attribute couldn't be found in the DataFrame's numeric attributes")
                return
        elif (type(attr) == list):
            searched_attrs = []
            for col in attr:
                if col.upper() in [col.upper() for col in self.num_columns]:
                    i = [num_col.upper() for num_col in self.num_columns].index(col.upper())
                    searched_attrs.append(self.num_columns[i])
                else:
                    print("The given attribute couldn't be found in the DataFrame's numeric attributes")
                    return
        else:
            print("The given attribute couldn't be found in the DataFrame's numeric attributes")
            return

        l_limit = []
        u_limit = []
        for col in self.num_columns:
            q1 = self.dataframe[col].quantile(0.25)
            q3 = self.dataframe[col].quantile(0.75)
            iqr = q3 - q1
            l_limit.append(q1 - 1.5 * iqr)
            u_limit.append(q3 + 1.5 * iqr)

        outliers_df = pd.DataFrame({"LowerLimit": l_limit, "UpperLimit": u_limit},
            index = self.num_columns) 
        self.outliers_range = outliers_df

        num_outliers_df = self.dataframe[searched_attrs].copy()
        for i, col in enumerate(searched_attrs):
            num_outliers_df.loc[:, col] = np.where(
                ((num_outliers_df.loc[:, col] < self.outliers_range.loc[col, "LowerLimit"]) |
                (num_outliers_df.loc[:, col] > self.outliers_range.loc[col, "UpperLimit"])) &
                (pd.notnull(num_outliers_df.loc[:, col])),
                num_outliers_df.loc[:, col], np.nan) 
    
        new_col = num_outliers_df.notna().sum(axis = 1).tolist()
        num_outliers_df["N_Outliers"] = new_col

        return num_outliers_df

        

