# ============================================================================
# PROJET ML - PRÉDICTION EURUSD
# ============================================================================

from KNN_and_Linear import run_baseline_models
from arima_model import run_arima_model

# ============================================================================
# EXÉCUTION DES MODÈLES BASELINE
# ============================================================================

if __name__ == "__main__":
    print("🚀 Exécution des modèles baseline (Linear Regression & KNN)\n")
    results = run_baseline_models()
    
    # Exécuter ARIMA (si disponible)
    arima_results = run_arima_model()
    
    print("\n" + "="*60)
    print("📊 PRÉDICTIONS DÉTAILLÉES")
    print("="*60)
    
    # Affichage des 10 premières prédictions
    print("\n10 premières prédictions vs valeurs réelles:")
    print(f"{'Index':<8} {'Réel':<12} {'Linear Reg':<12} {'KNN':<12}")
    print("-" * 50)
    for i in range(min(10, len(results['y_test']))):
        print(f"{i:<8} {results['y_test'].iloc[i]:<12.5f} {results['lr_predictions'][i]:<12.5f} {results['knn_predictions'][i]:<12.5f}")
    
    print("\n📈 MÉTRIQUES FINALES:")
    print(f"\nLinear Regression → RMSE: {results['lr_metrics']['RMSE']:.6f} | R²: {results['lr_metrics']['R²']:.4f}")
    print(f"KNN Regression    → RMSE: {results['knn_metrics']['RMSE']:.6f} | R²: {results['knn_metrics']['R²']:.4f}")
    if arima_results:
        print(f"ARIMA             → RMSE: {arima_results['metrics']['RMSE']:.6f} | R²: {arima_results['metrics']['R²']:.4f}")
    else:
        print("ARIMA             → non disponible (package manquant)")
    
    # Déterminer le meilleur modèle
    # Déterminer le meilleur modèle (par RMSE) parmi ceux disponibles
    best = ('Linear Regression', results['lr_metrics']['RMSE'])
    if results['knn_metrics']['RMSE'] < best[1]:
        best = ('KNN Regression', results['knn_metrics']['RMSE'])
    if arima_results and arima_results['metrics']['RMSE'] < best[1]:
        best = ('ARIMA', arima_results['metrics']['RMSE'])

    print(f"\n🏆 Meilleur modèle: {best[0]} (RMSE: {best[1]:.6f})")
    
    print("\n✅ Analyse terminée!")
    
    # Afficher les 5 premières lignes du forecast ARIMA si disponible
    if arima_results and 'forecast_df' in arima_results:
        print("\nARIMA - aperçu du forecast (5 premières lignes):")
        try:
            print(arima_results['forecast_df'].head(5).to_string(index=False))
        except Exception:
            # Si c'est un chemin ou une structure non-standard
            print("(Impossible d'afficher le forecast)")


