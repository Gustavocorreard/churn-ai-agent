    def generate_insights(feature_importance):
    
    top_features = feature_importance.head(5).index.tolist()
    
    insights = f"""
    Principais fatores que aumentam o churn:
    - {top_features[0]}
    - {top_features[1]}
    - {top_features[2]}
    
    Recomendação:
    Clientes com essas características devem ser priorizados em ações de retenção.
    """
    
    return insights