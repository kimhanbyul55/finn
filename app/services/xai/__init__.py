"""Explainability utilities for financial news sentiment analysis."""


def explain_sentiment(*args, **kwargs):
    from app.services.xai.lime_explainer import explain_sentiment as _explain_sentiment

    return _explain_sentiment(*args, **kwargs)

__all__ = ["explain_sentiment"]
