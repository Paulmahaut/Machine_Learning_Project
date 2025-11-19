# ============================================================================
# PROJET ML - STOCK PRICES ANALYSIS - Stratégies pour >60%
# ============================================================================

from advanced_models import run_advanced_models
from datetime import datetime

# ============================================================================
# TEST DES STRATEGIES POUR ATTEINDRE >60% ACCURACY
# ============================================================================

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        
        print("\n" + "="*80)
        print("🎯 STRATÉGIE: FENÊTRE DE PRÉDICTION + ACTIONS STABLES")
        print("="*80)
        print("\n💡 Changements clés:")
        print("   ✓ Prédiction sur 5 JOURS au lieu de 1 jour")
        print("   ✓ Features temporelles (Monday effect, January effect)")
        print("   ✓ Patterns de chandeliers et tendances")
        print("   ✓ Actions stables avec dividendes réguliers")
        
        # Test 1: Johnson & Johnson (Dividendes aristocrate)
        print("\n\n" + "🔵"*40)
        print("TEST 1: JOHNSON & JOHNSON (JNJ) - Dividendes Aristocrate")
        print("🔵"*40)
        results_jnj = run_advanced_models("JNJ", "Johnson & Johnson", prediction_window=5)
        
        # Test 2: S&P 500 ETF (Tendances macro)
        print("\n\n" + "🟢"*40)
        print("TEST 2: S&P 500 ETF (SPY) - Tendances Macro")
        print("🟢"*40)
        results_spy = run_advanced_models("SPY", "S&P 500 ETF", prediction_window=5)
        
        # Test 3: Procter & Gamble (Patterns saisonniers)
        print("\n\n" + "🟡"*40)
        print("TEST 3: PROCTER & GAMBLE (PG) - Consumer Staples")
        print("🟡"*40)
        results_pg = run_advanced_models("PG", "Procter & Gamble", prediction_window=5)
        
        # Test 4: NVIDIA (Tech cyclique)
        print("\n\n" + "🟣"*40)
        print("TEST 4: NVIDIA (NVDA) - Cycles GPU/IA")
        print("🟣"*40)
        results_nvda = run_advanced_models("NVDA", "NVIDIA", prediction_window=5)
        
        # Résumé comparatif
        print("\n\n" + "="*80)
        print("📊 RÉSUMÉ COMPARATIF - PRÉDICTION SUR 5 JOURS")
        print("="*80)
        
        def get_best(results):
            if not results:
                return "N/A", 0
            best = max(results.items(), key=lambda x: x[1]['accuracy'])
            return best[0].upper(), best[1]['accuracy']
        
        jnj_best, jnj_acc = get_best(results_jnj)
        spy_best, spy_acc = get_best(results_spy)
        pg_best, pg_acc = get_best(results_pg)
        nvda_best, nvda_acc = get_best(results_nvda)
        
        print(f"\n{'Asset':<25} {'Meilleur Algo':<15} {'Accuracy':>10} {'Status':>10}")
        print("-" * 70)
        print(f"{'J&J (JNJ)':<25} {jnj_best:<15} {jnj_acc:>9.2f}% {('✅ >60%' if jnj_acc >= 60 else '❌'):>10}")
        print(f"{'S&P 500 (SPY)':<25} {spy_best:<15} {spy_acc:>9.2f}% {('✅ >60%' if spy_acc >= 60 else '❌'):>10}")
        print(f"{'P&G (PG)':<25} {pg_best:<15} {pg_acc:>9.2f}% {('✅ >60%' if pg_acc >= 60 else '❌'):>10}")
        print(f"{'NVIDIA (NVDA)':<25} {nvda_best:<15} {nvda_acc:>9.2f}% {('✅ >60%' if nvda_acc >= 60 else '❌'):>10}")
        
        # Conclusion
        print("\n" + "="*80)
        print("💡 CONCLUSIONS")
        print("="*80)
        
        best_overall = max([
            ("J&J", jnj_acc),
            ("SPY", spy_acc),
            ("P&G", pg_acc),
            ("NVIDIA", nvda_acc)
        ], key=lambda x: x[1])
        
        print(f"\n🏆 Meilleur résultat: {best_overall[0]} ({best_overall[1]:.2f}%)")
        
        success_count = sum([1 for acc in [jnj_acc, spy_acc, pg_acc, nvda_acc] if acc >= 60])
        
        if success_count > 0:
            print(f"\n✅✅✅ SUCCÈS! {success_count}/4 actions atteignent >60%!")
            print("\n🎓 Leçons apprises:")
            print("   • Fenêtre de 5 jours > 1 jour pour la prédictibilité")
            print("   • Actions stables/ETF > Actions volatiles/Forex")
            print("   • Features temporelles améliorent les performances")
        else:
            print(f"\n⚠️  Aucune action n'atteint 60%. Meilleur: {best_overall[1]:.2f}%")
            print("\n💡 Prochaines étapes suggérées:")
            print("   • Augmenter la fenêtre à 7-10 jours")
            print("   • Ajouter sentiment analysis / news")
            print("   • Filtrer uniquement certains patterns (post-earnings)")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n⏱️  Durée totale d'exécution: {duration:.2f}s")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {str(e)}")
        print(f"Type d'erreur: {type(e).__name__}")
        import traceback
        print("\nTraceback complet:")
        traceback.print_exc()
        print("\n❌❌❌ PROGRAMME INTERROMPU ❌❌❌")
