mkdir -p /results

cat > /results/utils.py << 'EOF'
class ChurnPredictor:
    def __init__(self):
        pass

    def fit(self, train_df):
        print("Dummy fit")
        pass

    def predict(self, df):
        return np.zeros(len(df))

    def predict_proba(self, df):
        n = len(df)
        return np.column_stack([np.ones(n) * 0.3, np.ones(n) * 0.7])
EOF
