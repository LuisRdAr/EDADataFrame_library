import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dateutil.parser import parse
import math
import re
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

class EDADataFrame(pd.DataFrame):
    """
    The EDADataFrame class is built upon a pandas DataFrame and simplifies the 
    exploratory data analysis of its content through the inclusion of some new methods.
    Its initialization is the same as the pandas DataFrame one, although the target column
    can be optionally specified for machine learning purposes.
    All DataFrame's methods can be applied to EDADataFrame, returning an EDADataFrame.
    """

    @property
    def _constructor(self):
        return EDADataFrame


    def __init__(self, target: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)    
        _metadata = ["target", "cat_columns", "num_columns", "date_columns", "outliers_range"]
        self.target = target
        self.cat_columns = []
        self.num_columns = []
        self.date_columns = []


    def _set_dtype_classification(self, column: str, cat_threshold: float = 0.05) -> str:
        """
        Based on the column datatype, the number of unique and non-null values in it,
        classifies the attribute in numeric, categorical, boolean, text or date.

        Parameters
        ----------
        column: attribute in the EDADataFrame to be analyze.
        cat_threshold: ratio between the number of unique and non-null values.

        Returns
        ----------
        dtype: Datatype classified. Possibilities are: numeric, categorical, boolean, text or date
        
        """
        if column in self.columns:
            i = 0
            value = self.iloc[i].loc[column]
            while pd.isnull(value):
                i += 1
                value = self.iloc[i].loc[column]
            
            notnull = self[column].notnull().sum()/self.shape[0]
            n_unique = self[column].nunique()
            unique = n_unique/self.shape[0]
           
            if isinstance(value, str):
                try:
                    float(value)
                    dtype = "numeric"
                except:
                    try:
                        parse(value)
                        dtype = "date"
                    except:
                        if re.match("^[/[|/{]", value):
                            dtype = "structured"
                        elif unique / notnull < cat_threshold:
                            dtype = "categorical"
                        else:
                            dtype = "text"
            elif n_unique == 2:
                bool_values = self[column].value_counts().index.to_list()
                if (True in bool_values) | (False in bool_values):
                    dtype = "bool"
            else:
                try:
                    float(value)
                    dtype = "numeric"
                except:
                    dtype = "other"
        else:
            raise ValueError("Column {} is not defined in EDADataFrame object".format(column))
        
        return dtype 

    
    def general_info(self, cat_threshold: float = 0.05) -> pd.DataFrame:
        """
        Analyzes the EDADataFrame column by column and returns a pandas DataFrame
        containing the column datatype (primary), the datatype classification made 
        through _set_dtype_classification() (secondary datatype), number of non-null 
        values, percetange of non-null values in the column, number of unique values
        and percentage of unique values.
        Based on the the primary and secondary datatypes, the 'cat_columns' and 
        'num_columns' are filled.

        Parameters
        ----------
        cat_threshold: ratio between the number of unique and non-null values.

        Returns
        ----------
        df: pandas DataFrame containing the information relative to each column.

        """

        cols = self.columns.to_list()

        dtypes = []
        sec_dtypes = []
        notnulls = []
        notnulls_p = []
        unique = []
        unique_p = []

        cat_cols = []
        num_cols = []
        date_cols = []

        for col in cols:
            dtypes.append(str(self[col].dtype))
            notnulls.append(self[col].notnull().sum())
            notnulls_p.append(
                round(self[col].notnull().sum()/self.shape[0], 3))
            unique.append(self[col].nunique())
            unique_p.append(
                round(self[col].nunique()/self.shape[0], 3))  
            
            sec_dtypes.append(self._set_dtype_classification(col, cat_threshold))
            
            if ((dtypes[-1] == "object") & (sec_dtypes[-1] == "categorical" or sec_dtypes[-1] == "bool")) \
                | (dtypes[-1] == "bool"):
                cat_cols.append(col)
            elif ("int" in dtypes[-1] or "float" in dtypes[-1]):
                num_cols.append(col)
            #elif "date" in str(type(value)):
            #    date_cols.append(col) 

        df = pd.DataFrame({"Column": cols, "Detected_Dtype": dtypes, "Classified_Dtype": sec_dtypes,
                            "Non_Null": notnulls, "Non_Null_perc": notnulls_p, "Unique": unique, 
                            "Unique_perc": unique_p})
        
        print("DataFrame's shape: {} columns by {} rows" \
            .format(self.shape[1], self.shape[0]))
        #print(round(df,2), end = "\n\n")
        print("Number of incomplete rows (with one or more null values): {} out of {} ({}%)" \
            .format(self[self.isna().any(axis=1)].shape[0], 
                    self.shape[0], 
                    100*round(self[self.isna().any(axis=1)].shape[0]/self.shape[0], 4)))

        self.cat_columns = cat_cols
        self.num_columns = num_cols
        self.date_columns = date_cols

        print("Categorical Attributes: (Can be modified through method 'set_cat_attr')\n\t{}".format(self.cat_columns))
        print("Numerical Attributes: (Can be modified through method 'set_num_attr')\n\t{}".format(self.num_columns))
        print("Date Attributes: (Can be modified through method 'set_date_attr')\n\t{}".format(self.date_columns))

        return df

    
    def set_target(self, target: str) -> None:
        """
        Sets the target attribute of the EDADataFrame instance

         Parameters
        ----------
        target: target column of the EDADataFrame instance
        """

        if isinstance(new_cat_list, str) and target in self.columns:
            old_target = self.target
            self.target = target
            print("Previous target attributes:\n\t{}".format(old_target))
            print("Current target attributes:\n\t{}".format(self.target))
        else:
            raise ValueError("Column {} is not defined in EDADataFrame object".format(target))


    def set_cat_attr(self, new_cat_list: list) -> None:
        """
        Sets the cat_columns attribute of the EDADataFrame instance

         Parameters
        ----------
        new_cat_list: list which contains the categorical columns of the EDADataFrame instance
        """

        old_cat_columns = self.cat_columns
        if isinstance(new_cat_list, list):
            for attr in new_cat_list:
                if attr in self.columns:
                    pass
                else:
                    raise ValueError("Column {} is not defined in EDADataFrame object".format(attr))
                # i = 0
                # value = self.iloc[i].loc[column]
                # while pd.isnull(value):
                #     i += 1
                #     value = self.iloc[i].loc[column]
                # if type(value) == str:
                #     pass
                # else:
                #     raise ValueError("Categorical attributes must contain string values")
            self.cat_columns = new_cat_list
            print("Previous list of categorical attributes:\n\t{}".format(old_cat_columns))
            print("Current list of categorical attributes:\n\t{}".format(self.cat_columns))
        else:
            raise TypeError("The input must be a list")


    def set_num_attr(self, new_num_list: list) -> None:
        """
        Sets the num_columns attribute of the EDADataFrame instance

         Parameters
        ----------
        new_num_list: list which contains the numerical columns of the EDADataFrame instance
        """
        
        old_num_columns = self.num_columns
        if isinstance(new_num_list, list):
            for attr in new_num_list:
                if attr in self.columns:
                    pass
                else:
                    raise ValueError("Column {} is not defined in EDADataFrame object".format(attr))
            self.num_columns = new_num_list
            print("Previous list of numerical attributes:\n\t{}".format(old_num_columns))
            print("Current list of numerical attributes:\n\t{}".format(self.num_columns))
        else:
            raise TypeError("The input must be a list")

    
    def set_date_attr(self, new_date_list: list) -> None:
        """
        Sets the date_columns attribute of the EDADataFrame instance

         Parameters
        ----------
        new_date_list: list which contains the date columns of the EDADataFrame instance
        """

        old_date_columns = self.date_columns
        if isinstance(new_date_list, list):
            for attr in new_date_list:
                if attr in self.columns:
                    pass
                else:
                    raise ValueError("Column {} is not defined in EDADataFrame object".format(attr))
            self.num_columns = new_date_list
            print("Previous list of date attributes:\n\t{}".format(old_date_columns))
            print("Current list of date attributes:\n\t{}".format(self.date_columns))
        else:
            raise TypeError("The input must be a list")


    def search_duplicates(self, show = False) -> None or pd.DataFrame:
        """
        Searches for duplicated rows in the EDADataFrame, considering all columns, excluding each column
        and excluding each pair of columns.
        Aditionally, it can return a DataFrame containing the duplicated rows for its inspection.

        Parameters:
        ----------
        show: boolean variable that defines whether it returns the duplicated rows or not.

        Returns:
        ----------
        df: DataFrame with the duplicated rows.
        """

        print("This DataFrame has {} duplicated rows considering all its attributes and," \
            .format(self[self.duplicated(keep = False)].shape[0]))
        for i, col1 in enumerate(self.columns):
            n_duplicated = self[self.loc[:, self.columns != col1].duplicated(keep = False)].shape[0]
            print("\t {} duplicated rows when excluding only {},".format(n_duplicated, col1))
            for j, col2 in enumerate(self.columns):
                n_duplicated2 = self[self.loc[:, (self.columns != col1) & (self.columns != col2)] \
                    .duplicated(keep = False)].shape[0]
                if n_duplicated2 != 0:
                    print("\t\t {} duplicated rows when excluding the {}-{} pair," \
                        .format(n_duplicated2, col1, col2))


    def univariate_analysis(self, attrs: list = "All") -> None:
        """
        Plots barplots for the categorical attributes defined in self.cat_columns or
        specified categorical columns in 'attrs' arg, kdeplots and boxplots 
        for the numerical attributes in self.num_columns or specified numerical
        columns defined in 'attrs' arg.

        Parameters
        ----------
        attrs: by default it's assigned as "All". It can be defined as a list which 
        contains the columns desired to be plotted.
        """

        if self.cat_columns or self.num_columns:
            pass
        else:
            raise ValueError("cat_columns and num_columns attributes are empty")
        if attrs != "All":
            if isinstance(attrs, list):
                for attr in attrs:
                    if attr in self.columns:
                        pass
                    else:
                        raise ValueError("Column {} is not defined in EDADataFrame object".format(attr))
            else:
                raise TypeError("The input must be a list")
            
            cat_plots = [attr for attr in attrs if attr in self.cat_columns]
            num_plots = [attr for attr in attrs if attr in self.num_columns]

        else:
            cat_plots = self.cat_columns
            num_plots = self.num_columns

        n_cat_plots = len(cat_plots)
        if n_cat_plots > 0:
            rows_plots = math.ceil(n_cat_plots/3)
            fig, ax = plt.subplots(rows_plots, 3, figsize = (15, rows_plots*5), sharey = True)
            fig.suptitle('Univariate Analysis of the Categorical Attributes', fontsize=16)
            for i, col in enumerate(cat_plots):
                x = self[col].value_counts().index.to_list()
                y = self[col].value_counts(normalize = True).values
                sns.barplot(x = x, y = y, ax = ax[math.floor(i/3)][int(i%3)]).set_title(col)
                ax[math.floor(i/3)][int(i%3)].set_ylabel("Percentage") 
                
                for bar in ax[math.floor(i/3)][int(i%3)].patches:
                    ax[math.floor(i/3)][int(i%3)].annotate(format(bar.get_height(), '.3f'),
                            (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va='center',
                            size=10, xytext=(0, 5),
                            textcoords='offset points');
            plt.show()

        n_num_plots = len(num_plots)
        if n_num_plots > 0:
            rows_plots = math.ceil(n_num_plots/3)
            fig2, ax2 = plt.subplots(rows_plots, 3, figsize = (18, rows_plots*5))
            fig2.suptitle('Univariate Analysis of the Numerical Attributes', fontsize=16)
            for i, col in enumerate(num_plots):
                sns.kdeplot(data = self[num_plots], x = col, ax = ax2[math.floor(i/3)][int(i%3)])
                ax[math.floor(i/3)][int(i%3)].set_ylabel("Percentage") 
                
                for bar in ax[math.floor(i/3)][int(i%3)].patches:
                    ax[math.floor(i/3)][int(i%3)].annotate(format(bar.get_height(), '.3f'),
                            (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va='center',
                            size=10, xytext=(0, 5),
                            textcoords='offset points');
            plt.show()

            fig3, ax3 = plt.subplots(rows_plots, 3, figsize = (15, rows_plots*5))
            fig3.suptitle('Visualization of Outliers in Numerical Attributes', fontsize=16)
            for i, col in enumerate(num_plots):
                sns.boxplot(data = self[num_plots], x = col, ax = ax3[math.floor(i/3)][int(i%3)], fliersize = 0.5)
            plt.show()
    
    
    def correlation_matrix(self) -> None:
        """
        Under construction.
        Currently plots a heatmap containing the correlation matrix for numerical attributes.
        """

        sns.heatmap(self[self.num_columns].corr(), annot = True)
    

    def bivariate_analysis(self, x = "All", y = "All", graph = "") -> None:
        pass

    
    def outliers(self, attrs = "All") -> pd.DataFrame:
        """
        Calculates the limits for regular values in numeric columns and
        then searches for its outliers and marks them in an EDADataFrame.
        Outliers are shown with their actual values, whereas regular values
        are shown as NaNs.
        It can also search in a selected list of numerical columns.

        Parameters:
        ----------
        attrs: by default it's assigned as "All". It can be defined as a list which 
        contains the columns desired to be analyzed.
        """
        
        if self.num_columns:
            pass
        else:
            raise ValueError("num_columns attribute is empty")
        if attrs != "All":
            if isinstance(attrs, list):
                for attr in attrs:
                    if attr in self.columns:
                        pass
                    else:
                        raise ValueError("Column {} is not defined in EDADataFrame object".format(attr))
            else:
                raise TypeError("The input must be a list")
            
            searched_attrs = [attr for attr in attrs if attr in self.num_columns]
        else:
            searched_attrs = self.num_columns

        l_limit = []
        u_limit = []
        for col in self.num_columns:
            q1 = self[col].quantile(0.25)
            q3 = self[col].quantile(0.75)
            iqr = q3 - q1
            l_limit.append(q1 - 1.5 * iqr)
            u_limit.append(q3 + 1.5 * iqr)

        outliers_df = pd.DataFrame({"LowerLimit": l_limit, "UpperLimit": u_limit},
            index = self.num_columns) 
        self.outliers_range = outliers_df

        num_outliers_df = self[searched_attrs].copy()
        for i, col in enumerate(searched_attrs):
            num_outliers_df.loc[:, col] = np.where(
                ((num_outliers_df.loc[:, col] < self.outliers_range.loc[col, "LowerLimit"]) |
                (num_outliers_df.loc[:, col] > self.outliers_range.loc[col, "UpperLimit"])) &
                (pd.notnull(num_outliers_df.loc[:, col])),
                num_outliers_df.loc[:, col], np.nan) 
    
        new_col = num_outliers_df.notna().sum(axis = 1).tolist()
        num_outliers_df["N_Outliers"] = new_col

        return num_outliers_df

        

