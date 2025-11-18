# ============================================================================
# PROJET ML - PRÉDICTION EURUSD
# ============================================================================

from KNN_and_Linear import run_baseline_models

# ============================================================================
# EXÉCUTION DES MODÈLES BASELINE
# ============================================================================

if __name__ == "__main__":
    print("🚀 Exécution des modèles baseline (Linear Regression & KNN)\n")
    results = run_baseline_models()
    
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
    
    # Déterminer le meilleur modèle
    if results['lr_metrics']['RMSE'] < results['knn_metrics']['RMSE']:
        print("\n🏆 Meilleur modèle: Linear Regression")
    else:
        print("\n🏆 Meilleur modèle: KNN Regression")
    
    print("\n✅ Analyse terminée!")


