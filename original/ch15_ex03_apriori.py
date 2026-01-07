# ch15_ex03_apriori.py
# ch15.3 association rule mining

# load required modules
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# load data
df = pd.read_csv("data/ch15_transaction.csv")
X = df.groupby('id')['item'].apply(list)
print(X)

# Encode transaction data:
# Transform a list of transactions (i.e. a list of list of items) to a data frame 
# that a row represents a transaction and a column represents an item.
enc = TransactionEncoder()
X_transformed = enc.fit_transform(X)
df_transformed = pd.DataFrame(X_transformed, columns=enc.columns_)
print(df_transformed)

# ex15.3 - generate frequent itemsets with minimal support 0.4
# Use `apriori()` from `mlxtend.frequent_patterns` module.
frequent_items = apriori(df_transformed, min_support=0.4, use_colnames=True)
print(frequent_items)


# ex15.4 - find association rules with minimal confidence 0.7
# Use `association_rules()` from `mlxtend.frequent_patterns` module.
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.7)
print(rules[['antecedents', 'consequents', 'antecedent support', 'support', 'confidence', 'lift']])


