from hmmlearn.hmm import GaussianHMM

def train_hmm(df, n_states=3):
    """
    Train a Gaussian HMM on returns and volatility.
    Returns dataframe with regimes and trained model.
    """
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
    X = df[['return', 'volatility']].values
    model.fit(X)
    regimes = model.predict(X)
    df = df.copy()
    df['regime'] = regimes
    return df, model
