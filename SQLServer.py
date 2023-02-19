import pandas as pd
import pyodbc
import warnings


warnings.filterwarnings('ignore')


class SQLServer:
    def __init__(self, database):
        self.Driver = 'ODBC Driver 17 for SQL Server'
        self.Server = 'localhost'
        self.Database = database
        self.conn = pyodbc.connect(f'DRIVER={self.Driver};SERVER={self.Server};DATABASE={self.Database};Trusted_Connection=yes;')


    def select(self, query):
        return pd.read_sql(query, self.conn)
    