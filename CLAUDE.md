# CLAUDE.md — Directed-Evolution-loop

> Instructions pour Claude Code sur ce dépôt. Voir aussi `README.md` (setup, régénération des
> données, "Next tasks") — ce fichier-ci est un résumé d'orientation rapide + suivi de structure.

## ⚠ Consigne permanente

**À chaque création d'un nouveau fichier ou dossier significatif dans ce projet (nouveau notebook,
module `lib/`, sous-dossier), mettre à jour la section "Structure du projet" ci-dessous dans le même
tour.** Les données dérivées/gitignorées (`*.csv` de diversité, `*.npy`, `__pycache__`,
`.ipynb_checkpoints`, `.venv`) ne nécessitent pas d'entrée individuelle.

---

## Structure du projet

```
Directed-Evolution-loop/
├── Modelization_V1/                     # Modèle actif — tout le travail en cours se fait ici
│   ├── pyproject.toml                   # rend lib/ importable (sequence_classesV1, analysisV1, ...) depuis n'importe quel notebook
│   ├── lib/                             # modules partagés, importés "bare" par tous les notebooks
│   │   ├── sequence_classesV1.py        # moteur de simulation : Protocol/ProtocolV2/ProtocolV3, initialize_random_weights
│   │   ├── analysisV1.py                # helpers d'analyse/plot : pearson, precision_at_k, plot_teacher_weights, ...
│   │   ├── RegressionV1.py              # régression de Potts ridge-régularisée (F+J jointe) : sur NGS simulé
│   │   │                                #   (recover_weights_from_NGS) ou directement sur des données réelles
│   │   │                                #   (fit_weights_potts_from_data, cf. AAV9_potts_regression.ipynb)
│   │   ├── initialize_weights.py        # charge F_viab/J_viab AAV9 réels, construit F_sel/J_sel corrélés/anticorrélés/indép.
│   │   ├── MLP_regV1.py                 # tentative MLP profile-only, abandonnée
│   │   ├── aav9_{F,J}_viab_mlp.npy      # gitignorés — artefacts dérivés, régénérés via AAV9_profile_model.ipynb
│   │   └── cross_packaging_draft.py     # (2026-08-26) DRAFT non intégré, non importé ailleurs — sketch pour
│   │                                    #   modéliser le cross-packaging (variant non-viable co-transfecté avec un
│   │                                    #   variant fonctionnel dans la même cellule HEK, qui encapside par erreur
│   │                                    #   son ADN — explique la bimodalité observée sur fit4functionaav9.csv,
│   │                                    #   cf. log_enrichment_histograms.ipynb) : 2 sous-classes de ProtocolV3
│   │                                    #   redéfinissant produce_capsids(). Constat clé : Protocol.produce_capsids()
│   │                                    #   actuel tire C_s (cellules transfectées) INDÉPENDAMMENT par séquence — deux
│   │                                    #   séquences différentes ne partagent jamais une cellule simulée, donc
│   │                                    #   aucun substrat pour le cross-packaging n'existe dans le modèle actuel.
│   │                                    #   ProtocolCrossPackagingBackground (recommandée) : terme de fuite agrégé
│   │                                    #   (cross_packaging_rate * médiane des taux parmi les séquences
│   │                                    #   transfectées) ajouté au taux de Poisson de chaque séquence — reste O(d0),
│   │                                    #   testée manuellement (rate=0 reproduit exactement Protocol, rate>0 donne
│   │                                    #   un plancher non-nul à toutes les séquences au lieu de 99% à zéro).
│   │                                    #   Médiane plutôt que moyenne : la moyenne est tirée vers le haut par les
│   │                                    #   quelques séquences à score extrême (vérifié empiriquement, ~5x l'écart).
│   │                                    #   (note : une 1ère esquisse ProtocolCrossPackagingMechanistic — pool de
│   │                                    #   cellules physiques partagé, pas tractable à l'échelle N1 du projet — a été
│   │                                    #   retirée du fichier depuis, cf. historique git). Ajout (2026-08-26) d'une
│   │                                    #   3e source de bruit, DISTINCTE du cross-packaging (agit sur la mesure NGS,
│   │                                    #   pas sur la production) : LambdaWithHallucination (sous-classe de Lambda)
│   │                                    #   + ProtocolWithHallucination (sous-classe de ProtocolV3, branche
│   │                                    #   LambdaWithHallucination dans _ngs_and_deplete() à chaque checkpoint NGS).
│   │                                    #   Modélise l'apparition de nouveaux variants repérée dans
│   │                                    #   new_variant_appearance_analysis.ipynb (erreur PCR/synthèse qui fait
│   │                                    #   dériver l'ADN réel d'une construction loin de la séquence désignée) :
│   │                                    #   attributs `hallucination` (bool, défaut False) et `hallucination_rate`
│   │                                    #   (défaut 69/74464≈0.000927, la fraction empirique mesurée dans ce
│   │                                    #   notebook), réglables après construction comme `multinomialNGS`
│   │                                    #   (`lambda_obj.hallucination = True`). sequence_reads() ajoute
│   │                                    #   Poisson(D/d0) reads à chaque séquence tirée Bernoulli(hallucination_rate),
│   │                                    #   indépendamment de son abondance réelle — la déplétion reste basée sur les
│   │                                    #   molécules réellement pipettées, pas sur les reads hallucinés. Testée
│   │                                    #   manuellement via un round complet de loop_DE (poids construits à la main,
│   │                                    #   PAS via initialize_random_weights() — cf. note ci-dessous). Complétée
│   │                                    #   (2026-08-26) par `ProtocolCrossPackagingAndHallucination` (hérite des deux
│   │                                    #   `ProtocolCrossPackagingBackground` + `ProtocolWithHallucination` à la fois,
│   │                                    #   héritage multiple coopératif sans conflit — chaque classe redéfinit une
│   │                                    #   étape différente du pipeline) ; utilisée dans
│   │                                    #   `aav_viability_test/aav9/AAV9_cross_packaging_and_hallucination_impact.ipynb`.
│   ├── notebooks/                       # (réorganisé 2026-08-27 : "analysis of correlation"/"analysis of
│   │                                    #   parameters for viability"/deeper_mlp fusionnés dans
│   │                                    #   viability_parameter_sweeps/ ; "reproductibility" corrigé en
│   │                                    #   reproducibility/ ; plus aucun nom de dossier avec espace)
│   │   ├── viability_parameter_sweeps/  # Protocol_parameters_and_first_classic_use.ipynb ; mu_HEK_multiplicity_sweep.ipynb
│   │   │                                #   (mu = rho*N1/d0, HEK cells transfectées/séquence) : sweep de mu (F_viab/J_viab AAV9
│   │   │                                #   réel, F_sel/J_sel permutés, pool fixe 20k) ; sections 1-6 corrélation GT vs target1 ;
│   │   │                                #   section 7 ajoute un ProfileMLP entraîné sur un split train/test du pool, recovery du
│   │   │                                #   top-1000 (GT<->protocole, GT<->MLP, protocole<->MLP) au mu baseline (7.1-7.5) puis en
│   │   │                                #   fonction de mu (7.6, un MLP par mu) + comparaison meilleur/pire mu (7.7) ; section 8
│   │   │                                #   répète 7.6-7.7 avec noise_viab=noise_sel=0 pour isoler l'effet du sous-échantillonnage
│   │   │                                #   lié à mu de celui du bruit ajouté explicitement
│   │   │                                # T_viab_sweep.ipynb : même recette (pool 20k, split train/test, ProfileMLP, topk_recovery)
│   │   │                                #   mais mu FIXÉ à 50 et sweep de T_viab (0.2 à 50, "température" qui contrôle la
│   │   │                                #   netteté de exp(score/T_viab) dans produce_capsids) ; Pearson r et recovery top-1000
│   │   │                                #   (GT/protocole/MLP) tous deux unimodaux en T_viab, avec un pic pas forcément au même
│   │   │                                #   T_viab pour les deux métriques (r global vs recovery sur la seule queue extrême)
│   │   │                                # noise_viab_sweep.ipynb : mu=50 ET T_viab=0.8 fixés (valeurs trouvées dans les deux
│   │   │                                #   notebooks précédents), pool BEAUCOUP plus grand (d0=200 000 au lieu de 20 000,
│   │   │                                #   batch_size relevé à 2048 en conséquence), sweep de noise_viab (bruit d'expression
│   │   │                                #   par cellule) ; teste si le ProfileMLP débruite réellement (GT<->MLP doit rester
│   │   │                                #   au-dessus de GT<->protocole quand le bruit augmente) ; multinomialNGS=True (nouveau
│   │   │                                #   défaut standard du projet, cf. ci-dessous). Section 10 (ajoutée après coup) : sweep
│   │   │                                #   2D d0 x noise_viab — pour chaque noise_viab (grille 1D existante), 5 valeurs de d0
│   │   │                                #   (20k/50k/100k/200k/500k, incluant le d0=200k de base du notebook) ; boucle externe
│   │   │                                #   sur d0 (pool + ProtocolV3 reconstruits à neuf, N1=mu*d0/rho pour garder mu=50),
│   │   │                                #   boucle interne sur noise_viab (objet protocol réutilisé, comme le sweep 1D) ; 40
│   │   │                                #   combinaisons (d0, noise_viab), chacune avec un ProfileMLP frais ; recovery top-1000
│   │   │                                #   tracée à la fois sur le test fold interne au sweep et sur le pool d'éval fixe 50k
│   │   │                                # D_sequencing_depth_sweep.ipynb : même mu=50/T_viab=0.8/noise_viab=0.5/d0=200 000 fixés,
│   │   │                                #   sweep de D (profondeur de séquençage NGS par checkpoint, Multinomial(D, proportions)
│   │   │                                #   si multinomialNGS=True) de 1e6 à 1e10 ; courbes attendues monotones croissantes
│   │   │                                #   (contrairement à T_viab, pas de régime "trop élevé" pathologique), avec plateau de
│   │   │                                #   rendements décroissants à D élevé une fois les autres sources de bruit dominantes
│   │   │                                # diversity_sweep.ipynb : cette fois d0 (diversité de la librairie, 5k à 200k) EST la
│   │   │                                #   variable balayée ; mu=10 compensé à chaque d0 via N1=mu*d0/rho (rho=1e-4 fixé,
│   │   │                                #   10x plus petit que RHO_REF ailleurs — sans incidence par elle-même, cf. section 3
│   │   │                                #   de mu_HEK), mais D=1e8 reste FIXE (non compensé) donc reads/séquence=D/d0 diminue
│   │   │                                #   avec d0 : isole l'effet "dilution d'un budget NGS fixe par une librairie plus
│   │   │                                #   diverse", indépendamment de l'effet multiplicité déjà couvert par mu_HEK. Pool +
│   │   │                                #   ProtocolV3 reconstruits à neuf à chaque d0 (contrairement aux autres sweeps qui
│   │   │                                #   mutent un seul objet protocol réutilisé) ; batch_size adaptatif (max(256,
│   │   │                                #   n_train//50)) vu l'écart de taille de pool entre le plus petit et le plus grand
│   │   │                                #   d0 testé. N0=150*N1 (contrainte labo : N1 doit rester au moins 150x plus petit
│   │   │                                #   que N0) — c'est cette contrainte, combinée à mu, qui plafonne la grille à d0=200k
│   │   │                                #   (grille allait jusqu'à 1M sous l'ancien défaut mu=50, ce qui poussait N0 au-delà
│   │   │                                #   du plafond pratique ~5e12 ; baissé à mu=10 pour cette famille de notebooks plutôt
│   │   │                                #   que de garder mu=50 avec une grille tronquée — cf. "État actuel")
│   │   │                                # diversity_sweep_adaptive_D.ipynb : compagnon direct du précédent, mêmes mu/rho/T_viab/
│   │   │                                #   noise_viab/grille de d0/N0=150*N1, mais D(d0) = 50 000 * d0 (reads/séquence CONSTANT
│   │   │                                #   à 50 000, soit le ratio du baseline D=1e9/d0=20 000 déjà utilisé partout ailleurs —
│   │   │                                #   pas une valeur externe arbitraire) au lieu de D fixe : isole si la diversité a un
│   │   │                                #   coût propre au-delà de la simple dilution du budget NGS déjà montrée dans diversity_sweep
│   │   │                                # unified_parameter_sweep.ipynb (2026-08-24) : consolide TOUS les sweeps ci-dessus
│   │   │                                #   (mu, T_viab, noise_viab 1D+2D, D, diversity D-fixe/D-adaptatif) PLUS la comparaison
│   │   │                                #   shallow/deep MLP de deeper_mlp/, dans UN seul notebook prévu pour tourner sans
│   │   │                                #   surveillance (nuit) — but explicite : une base de paramètres partagée (rho=1e-4,
│   │   │                                #   mu=10, N0=150*N1, D=1e9, T_viab=0.8, noise_viab=0.5, pool fixe d0=20 000) réutilisée
│   │   │                                #   par chaque section sauf le paramètre balayé, là où les notebooks séparés avaient
│   │   │                                #   dérivé vers des bases incohérentes (mu 25-50, rho 1e-3/1e-4, tailles de pool
│   │   │                                #   20k/30k/50k/200k selon le notebook). Sections A-D (mu/T_viab/noise_viab/D) partagent
│   │   │                                #   LE MÊME objet ProtocolV3 réutilisé en séquence (comme dans chaque notebook source) ;
│   │   │                                #   sections E-H (2D noise_viab×d0, diversity D-fixe, diversity D-adaptatif, shallow vs
│   │   │                                #   deep) reconstruisent un pool frais par d0, avec DIVERSITY_GRID (5k-200k) partagé
│   │   │                                #   partout. Corrige au passage l'incohérence de deeper_mlp/diversity_sweep_deeper_mlp.ipynb
│   │   │                                #   (D=5e8/noise_viab=3 au lieu de D=1e8/noise_viab=0.5) — section H utilise maintenant
│   │   │                                #   exactement les mêmes valeurs que la section F, rendant la comparaison shallow/deep
│   │   │                                #   réellement valide. Les notebooks sources individuels restent en place, NON modifiés
│   │   │                                #   par ce fichier — mu_HEK_multiplicity_sweep.ipynb/T_viab_sweep.ipynb/noise_viab_sweep.ipynb/
│   │   │                                #   D_sequencing_depth_sweep.ipynb gardent leur propre base (mu=25-50, rho=1e-3, d0
│   │   │                                #   20k/30k/200k selon le fichier) ; seuls diversity_sweep.ipynb/diversity_sweep_adaptive_D.ipynb/
│   │   │                                #   diversity_sweep_deeper_mlp.ipynb ont été mis à jour séparément vers mu=10/N0=150*N1
│   │   │                                #   (cf. "État actuel" 2026-08-24) — la base harmonisée n'existe que dans ce nouveau fichier
│   │   │                                # (déplacé ici 2026-08-27, ex-"analysis of correlation/") F_permutation_recovery_correlation.ipynb :
│   │   │                                #   F_viab AAV9 réel, J_viab=J_sel=0 (pas d'épistasie pour ce premier passage) ; F_sel = 10
│   │   │                                #   permutations des lignes (axe acide aminé) de F_viab (une par clé jax), corrélation GT avec
│   │   │                                #   F_viab mesurée ; pool de séquences + split train/val/test fixes pour les 10 runs ; un
│   │   │                                #   ProfileMLP entraîné par permutation sur le log enrichment de sélectivité (target2), F_sel_hat
│   │   │                                #   recouvré par scan single-mutant, comparé au F_sel vrai de ce run
│   │   │                                # (déplacé ici 2026-08-27, ex-deeper_mlp/) diversity_sweep_deeper_mlp.ipynb : proche de
│   │   │                                #   diversity_sweep.ipynb mais PAS identique — mu=10/rho=1e-4/N0=150*N1 alignés, mais D=5e8 fixe
│   │   │                                #   (pas 1e8) et noise_viab=3 (pas 0.5, écart non résolu — cf. "État actuel") ; grille d0 propre
│   │   │                                #   200 à 200k (sur-ensemble du 5k-200k de diversity_sweep.ipynb côté petit d0) mais compare
│   │   │                                #   CETTE FOIS deux architectures entraînées sur les mêmes données : ShallowProfileMLP (~27k
│   │   │                                #   params, l'archi standard du projet, juste renommée) vs DeepProfileMLP (~90k params, nouveau)
│   │   │                                #   — embedding par position partagé (Linear 20->16 + GELU) moyenné (avg pool) sur les 7
│   │   │                                #   positions pour la branche "profile", + branche pairwise (MLP sur les 21 paires de positions
│   │   │                                #   concaténées, avg pool) plus expressive que le BilinearHead déjà testé (et jugé marginal) dans
│   │   │                                #   MLP_bilinear_head_anticorrelated.ipynb, + tête dense à 4 couches (256-128-64-32) au lieu de 2 ;
│   │   │                                #   même batch_size adaptatif pour isoler l'architecture comme seule variable ; inclut les mêmes
│   │   │                                #   diagnostics d'entraînement (best_epoch/total_steps/val_mse) que diversity_sweep.ipynb pour
│   │   │                                #   vérifier si le modèle profond profite réellement de plus de gradient steps, pas seulement de
│   │   │                                #   plus de capacité. Recovery top-1000 comparée sur le pool d'éval fixe avec DEUX références
│   │   │                                #   GT<->protocole : profondeur fixe (constante ~93%, section 4) et profondeur APPARIÉE
│   │   │                                #   (D_eval_matched = D_FIXED/d0 * 50 000, ré-simulée à chaque d0 — le vrai point de comparaison
│   │   │                                #   "juste", puisque les labels d'entraînement du MLP sont eux-mêmes à cette profondeur
│   │   │                                #   D_FIXED/d0, pas à la profondeur fixe du pool d'éval) ; graphe recovery sur le split interne au
│   │   │                                #   sweep supprimé (sa taille dépend de d0, donc pas comparable d'un point à l'autre)
│   │   │                                # GT basculée vers Potts (2026-08-27, cf. "État actuel") sur les 10 notebooks de cette
│   │   │                                #   section (aucun exclu — AAV9_fitting_protocol.ipynb, seul notebook non basculé du projet,
│   │   │                                #   est dans aav_viability_test/aav9/, pas ici) ; sorties de cellules effacées, caches
│   │   │                                #   diversity*.csv obsolètes supprimés, à ré-exécuter avant de faire confiance à un chiffre
│   │   │                                #   affiché. Caveat particulier sur mu_HEK_multiplicity_sweep.ipynb/unified_parameter_sweep.ipynb :
│   │   │                                #   tous deux réutilisent le point de fonctionnement "réaliste" (mu=50/T_viab=0.8/noise_viab=0.5/
│   │   │                                #   D=1e9) dérivé dans AAV9_fitting_protocol.ipynb CONTRE L'ANCIENNE GT — ce point de
│   │   │                                #   fonctionnement n'a pas été re-dérivé contre la GT Potts (choix utilisateur, cf. plus bas).
│   │   ├── directed_evolution_loop/     # DE_loopV1.ipynb — boucle de simulation d'évolution dirigée bout-en-bout
│   │   ├── selectivity_weight_regimes/  # trio F_sel/J_sel corrélé/anticorrélé/indépendant + variantes (bilinear head,
│   │   │                                #   double mutant designed, profile-only) + CSV de diversité mis en cache
│   │   │                                #   MLP_viability_noise_denoising20K.ipynb / MLP_viability_noise_denoising50K.ipynb
│   │   │                                #   (2 fichiers, pas 1 — pool de 20k/50k séquences respectivement) : viabilité
│   │   │                                #   SEULE (GT F_viab/J_viab régularisé), sweep de noise_viab à pool de séquences
│   │   │                                #   fixe pour tester si le MLP débruite (corrélation prédiction vs score vrai vs.
│   │   │                                #   label NGS brut)
│   │   │                                # GT basculée vers Potts (2026-08-27, cf. "État actuel") sur les 11 notebooks de
│   │   │                                #   ce dossier qui chargent F_viab/J_viab AAV9 (les 2 ci-dessus + cheated_library_MLP/
│   │   │                                #   initialize_weights_playground/MLP_bilinear_head_anticorrelated/MLP_designed_double_
│   │   │                                #   mutant_sampling/MLP_for_{anti,}correlated_weights/MLP_for_independent_weights/
│   │   │                                #   MLP_for_profile_only_weights/plotting_realistic_weights.ipynb) — sorties de
│   │   │                                #   cellules effacées, caches diversity*.csv obsolètes supprimés (dont le dossier
│   │   │                                #   `_stale_true_lambda_csv_backup/` laissé tel quel, déjà marqué obsolète avant ce
│   │   │                                #   changement), à ré-exécuter avant de faire confiance à un chiffre affiché. Plusieurs
│   │   │                                #   de ces notebooks citent en prose des chiffres précis (magnitudes, percentiles)
│   │   │                                #   calculés sous l'ANCIENNE GT — pas corrigés dans le texte, seulement dans le code.
│   │   ├── aav_viability_test/          # AAV5_profile_model.ipynb (entraîne ProfileMLP sur données réelles),
│   │   │                                #   checks de recouvrement d'erreur/top500 (AAV2_profile_model.ipynb déplacé
│   │   │                                #   dans obsolete/, cf. plus bas)
│   │   │   └── aav9/                    # (créé 2026-08-31, déplacement manuel de l'utilisateur) tous les 7 notebooks AAV9 + aav9.csv
│   │   │                                # AAV9_fitting_protocol.ipynb (2026-08-25) : SEUL notebook du projet volontairement
│   │   │                                #   PAS basculé vers la GT Potts (2026-08-27, décision utilisateur) — reste le
│   │   │                                #   document historique de comment mu/T_viab/noise_viab/D ont été dérivés (via
│   │   │                                #   recherche contre R_REAL = pearson(target réel, score GT) calculé sous
│   │   │                                #   l'ANCIENNE GT naïve) ; le retoucher sans re-lancer sa partie 2 laisserait des
│   │   │                                #   résultats affichés incohérents avec le code. AAV9_potts_GT_score_study.ipynb
│   │   │                                #   (2026-08-27, cf. plus bas) est son pendant côté GT Potts : réutilise CE MÊME
│   │   │                                #   point de fonctionnement sans le re-dériver. Calibre les hyperparamètres du
│   │   │                                #   protocole simulé (mu/T_viab/noise_viab/D) pour imiter la stochasticité du
│   │   │                                #   VRAI aav9.csv (68 776 séquences), pas juste recouvrer le score GT. Partie 1 :
│   │   │                                #   5 runs identiques du protocole (même F_viab/J_viab réels AAV9), histogramme
│   │   │                                #   des 10 corrélations pairwise → plafond de repeatability intrinsèque au
│   │   │                                #   protocole simulé (r≈0.99 au baseline mu=50/T_viab=0.8/noise_viab=0.5/D=1e9,
│   │   │                                #   donc quasi pas de bruit à ce point de fonctionnement, sur cette librairie
│   │   │                                #   réelle à large dynamique). Partie 2 : `R_REAL = pearson(real_target,
│   │   │                                #   viab_score_GT)` (viab_score_GT = compute_score(F_viab,J_viab), déterministe,
│   │   │                                #   sans bruit protocole) sert de cible — recherche aléatoire (pas grille
│   │   │                                #   complète, 10 800 combos impraticable) sur (mu,T_viab,noise_viab,D) pour
│   │   │                                #   qu'un run simulé atteigne r(sim,GT)≈R_REAL dans ~50% des répétitions
│   │   │                                #   (= médiane), pas en collant à real_target directement (hors d'atteinte, cf.
│   │   │                                #   plafond partie 1). Classement des candidats par `|r_mean - R_REAL|` (PAS par
│   │   │                                #   success_fraction seule — avec peu de répétitions elle est trop grossière,
│   │   │                                #   quasi binaire 0/1, et peut désigner arbitrairement un mauvais candidat en cas
│   │   │                                #   d'égalité — bug corrigé en cours de route). Diagnostic clé : un bon `r`
│   │   │                                #   global peut être trompeur sur une distribution bimodale (sépare juste les
│   │   │                                #   deux clusters viable/mort) sans imiter l'étalement réel à l'intérieur de
│   │   │                                #   chaque cluster (pics simulés trop étroits/déterministes si T_viab est bas —
│   │   │                                #   T_viab amplifie l'écart de score SANS diviser le terme de bruit, donc SNR
│   │   │                                #   explose à T_viab bas) — d'où la grille de tous les candidats testés
│   │   │                                #   (histogrammes réel vs simulé côte à côte, `frac(target1=0)` affiché : un
│   │   │                                #   score correct peut aussi venir d'une majorité de séquences invisibles aux
│   │   │                                #   deux checkpoints NGS, `target1=log(1)=0` par construction du pseudocount).
│   │   │                                #   Partie 5 : cellule manuelle éditable (hyperparamètres + F_viab/J_viab) pour
│   │   │                                #   explorer à la main. Partie 6 : superposition score GT vs log enrichment
│   │   │                                #   simulé (ATTENTION : pas la même échelle si T_viab≠1, E[target1]≈score/T_viab
│   │   │                                #   pas score — le Pearson r n'est lui pas affecté, invariant à un rescaling
│   │   │                                #   positif). Partie 7 : ProfileMLP (même archi que AAV9_profile_model.ipynb)
│   │   │                                #   entraîné sur 25 000 séquences réelles, testé sur le reste, histogrammes
│   │   │                                #   réel vs prédit superposés. `error` dans aav9.csv est une constante 0.1
│   │   │                                #   partout (placeholder, pas une vraie incertitude par séquence) — pas de vraie
│   │   │                                #   structure de réplicats dans ce CSV pour calibrer le bruit autrement.
│   │   │                                # AAV2_fitting_protocol.ipynb / AAV5_fitting_protocol.ipynb (2026-08-25) :
│   │   │                                #   sections 5-7 d'AAV9_fitting_protocol.ipynb rejouées sur aav2.csv (53 383
│   │   │                                #   séquences) et aav5.csv (737 588 séquences) — PAS les parties 1/3/4 (repeatability
│   │   │                                #   + recherche aléatoire, ~11 min à elle seule sur aav9, pas rejouée deux fois de
│   │   │                                #   plus) ; mêmes hyperparamètres baseline fixes (mu=50/T_viab=0.8/noise_viab=0.5/
│   │   │                                #   D=1e9) que aav9, non re-tunés par dataset, donc D/d0 (reads/séquence) diffère
│   │   │                                #   fortement entre les 3 notebooks vu les tailles de librairie très différentes.
│   │   │                                #   F_viab/J_viab chargés directement via np.load (pas de load_F_viab_aavX_mlp()
│   │   │                                #   générique dans initialize_weights.py, module spécifique à aav9 par design).
│   │   │                                #   Différence notable repérée : contrairement à aav9.csv, aav2.csv/aav5.csv ont
│   │   │                                #   une colonne `error` RÉELLE (variable par séquence, pas une constante 0.1) et
│   │   │                                #   des colonnes de comptages bruts plasmid/vector(1/2) — potentiellement de quoi
│   │   │                                #   calibrer le bruit réel plus directement que pour aav9 (non exploité dans ces
│   │   │                                #   deux notebooks, qui ne font que rejouer 5-7 à l'identique — piste ouverte).
│   │   │                                # AAV9_cross_packaging_and_hallucination_impact.ipynb (2026-08-26) : reprend les
│   │   │                                #   parties 0-2b (setup, données réelles, poids GT, viab_score_GT, R_REAL) et
│   │   │                                #   5-6 (playground manuel, target1 simulé vs real target, vs GT) d'
│   │   │                                #   AAV9_fitting_protocol.ipynb — SANS les parties 1 (repeatability) et 2
│   │   │                                #   (recherche aléatoire d'hyperparamètres) ni la partie 7 (ProfileMLP sur
│   │   │                                #   données réelles, indépendante du choix de Protocol). Même config baseline
│   │   │                                #   (mu=50/T_viab=0.8/noise_viab=0.5/D=1e9) passée à 4 variantes de
│   │   │                                #   lib/cross_packaging_draft.py : ProtocolV3 nu, ProtocolCrossPackagingBackground
│   │   │                                #   (cross_packaging_rate=0.05, valeur d'illustration non calibrée),
│   │   │                                #   ProtocolWithHallucination (hallucination_rate=69/74464, valeur empirique de
│   │   │                                #   new_variant_appearance_analysis.ipynb), et la nouvelle
│   │   │                                #   ProtocolCrossPackagingAndHallucination (les deux combinées — héritage multiple
│   │   │                                #   coopératif sur les 2 classes existantes, aucun conflit car elles redéfinissent
│   │   │                                #   des étapes différentes du pipeline : produce_capsids() vs _ngs_and_deplete()).
│   │   │                                #   Résultat : le cross-packaging seul fait descendre r(sim,real) de 0.861 à
│   │   │                                #   0.854, quasiment pile sur R_REAL=0.853 (le plafond de fidélité de la vraie
│   │   │                                #   expérience) ; l'hallucination seule a un impact négligeable à son taux
│   │   │                                #   empirique (0.093%, trop faible pour bouger la métrique à cette échelle) ;
│   │   │                                #   les deux combinées ≈ cross-packaging seul (hallucination toujours négligeable).
│   │   │                                #   GT basculée vers Potts (2026-08-27, cf. "État actuel") — sorties de cellules
│   │   │                                #   effacées, à ré-exécuter ; le chiffre R_REAL=0.853 cité ci-dessus a été calculé
│   │   │                                #   sous l'ANCIENNE GT, pas mis à jour dans ce texte (mu/T_viab/noise_viab/D restent
│   │   │                                #   le point de fonctionnement dérivé dans AAV9_fitting_protocol.ipynb contre
│   │   │                                #   l'ancienne GT, non re-dérivé — cf. plus bas).
│   │   │                                # AAV9_potts_regression.ipynb (2026-08-27) : remplace le duo "moyennes naïves
│   │   │                                #   par cellule (F) + résidu naïf par cellule ensuite rétréci indépendamment
│   │   │                                #   (J, empirique-Bayes/James-Stein, section 3c d'AAV9_profile_model.ipynb)"
│   │   │                                #   par UN SEUL fit de régression de Potts jointe et ridge-régularisée (F et J
│   │   │                                #   résolus ensemble, pas F d'abord puis J en résidu) directement sur aav9.csv,
│   │   │                                #   via `lib/RegressionV1.fit_weights_potts_from_data` (nouvelle fonction,
│   │   │                                #   ajoutée dans cette même passe — jusqu'ici RegressionV1.py ne fittait que
│   │   │                                #   sur les lectures NGS multi-round d'un `Protocol` SIMULÉ via
│   │   │                                #   `recover_weights_from_NGS`, jamais sur des données réelles). Design
│   │   │                                #   (single-site one-hot + pairwise outer-product + biais, 8 541 features) et
│   │   │                                #   ridge CV (grille `np.logspace(-1, 2, 30)`, 5-fold) réutilisés tels quels ;
│   │   │                                #   `sample_weight` optionnel ajouté à `ridge_cv_mse_potts`/`fit_weights_potts`
│   │   │                                #   (rétrocompatible, `None` par défaut) pour une réutilisation future sur
│   │   │                                #   aav2.csv/aav5.csv (colonne `error` réelle, contrairement à aav9.csv) — non
│   │   │                                #   exploité dans ce notebook. Résultats (68 776 séquences, meilleur
│   │   │                                #   lambda=23.95, ni au plancher ni au plafond de la grille) : design
│   │   │                                #   rang-déficient même à n > p (7715/8541 — beaucoup de cellules (i,j,a,b)
│   │   │                                #   jamais co-observées dans la vraie librairie combinatoire), d'où l'intérêt
│   │   │                                #   réel de la ridge, pas juste une précaution. `F_potts` quasi identique à
│   │   │                                #   `F_viab_GT` (r=+0.997, F était déjà bien estimé par les group-means, ~490
│   │   │                                #   séquences de support/cellule) ; `J_potts` diverge plus de `J_naive_final`
│   │   │                                #   (r=+0.839, exactement là où le group-means séquentiel + shrinkage
│   │   │                                #   par-cellule est structurellement le plus faible). Test prédictif tenu à
│   │   │                                #   l'écart (absent jusqu'ici pour la GT naïve — jamais validée par un vrai
│   │   │                                #   split train/test sur données réelles), même split qu'
│   │   │                                #   AAV9_profile_model.ipynb (`test_size=0.5, random_state=0`), les deux
│   │   │                                #   méthodes fittées sur `idx_train` seul : Pearson r sur `idx_test` réel =
│   │   │                                #   0.7821 (naïve, coupure dure `min_support=5`) vs 0.8467 (Potts
│   │   │                                #   régression) — nette amélioration de généralisation. Test de crédibilité
│   │   │                                #   (reprend le diagnostic brute-force de 3c, 2M séquences aléatoires
│   │   │                                #   uniformes) : nuance importante, PAS une victoire nette dans
│   │   │                                #   cette direction — le top-500 de la régression Potts s'appuie
│   │   │                                #   proportionnellement PLUS sur `J_part` que celui de la GT naïve actuelle
│   │   │                                #   (ratio J_part/F_part moyen du top-500 : 0.885 pour Potts contre 0.148
│   │   │                                #   pour la GT naïve déjà atténuée à REG_STRENGTH=5) — attendu, puisque le
│   │   │                                #   lambda choisi par CV optimise la prédiction sur des séquences de la
│   │   │                                #   distribution réelle, pas la plausibilité d'une extrapolation à des
│   │   │                                #   combinaisons aléatoires uniformes ; un lambda plus élevé (au prix d'un peu
│   │   │                                #   de r prédictif) reste à explorer si ce compromis compte pour l'usage en
│   │   │                                #   aval. Exporte `lib/aav9_F_viab_potts.npy`/`aav9_J_viab_potts.npy`, chargés
│   │   │                                #   par `initialize_weights.load_F_viab_aav9_potts`/`load_J_viab_aav9_potts`
│   │   │                                #   (nouveau, à côté de `load_F_viab_aav9_mlp`/`load_J_viab_aav9_mlp` — PAS un
│   │   │                                #   remplacement, aucun notebook existant n'a changé d'import ; devenir la GT
│   │   │                                #   par défaut impliquerait de re-caler mu/T_viab/noise_viab/D dans
│   │   │                                #   AAV9_fitting_protocol.ipynb, pas fait ici). Section 5b ajoutée après coup :
│   │   │                                #   sensibilité au choix de lambda — refit F/J (données complètes) aux 5 lambdas
│   │   │                                #   de la grille CV dont le MSE est le plus proche du minimum (23.95, MSE=4.703),
│   │   │                                #   histogrammes superposés du score GT résultant. Résultat rassurant : les 5
│   │   │                                #   candidats (11.7 à 30.4, tous à <0.5% du MSE minimum) donnent des scores GT
│   │   │                                #   quasi identiques (r≥0.9996 vs le lambda choisi) — le GT n'est pas sensible au
│   │   │                                #   choix précis du lambda dans la zone plate du MSE de CV.
│   │   │                                # AAV9_potts_GT_score_study.ipynb (2026-08-27) : rejoue l'étude de score GT
│   │   │                                #   d'AAV9_fitting_protocol.ipynb (section 2b : distribution du score GT
│   │   │                                #   déterministe ; sections 5-6 : log enrichment simulé par le Protocol vs
│   │   │                                #   score GT vs target réel) en remplaçant la source de GT par F_potts/J_potts
│   │   │                                #   (chargés via les loaders `load_F_viab_aav9_potts`/`load_J_viab_aav9_potts`,
│   │   │                                #   PAS re-fittés — la régression ridge elle-même reste dans
│   │   │                                #   AAV9_potts_regression.ipynb), mêmes hyperparamètres baseline QUE
│   │   │                                #   la section 2 d'AAV9_fitting_protocol.ipynb SAUF T_viab
│   │   │                                #   (mu=50/T_viab=1.3/noise_viab=0.5/D=1e9/RHO_REF=1e-3) — 1.3 est la
│   │   │                                #   température de base propre à la GT Potts (2026-08-31, précisé par
│   │   │                                #   l'utilisateur), PAS 0.8 (qui reste la valeur de l'ancienne GT naïve).
│   │   │                                #   Ajoute aussi une
│   │   │                                #   section absente ailleurs : quelques `J_potts[i,j]` individuels affichés
│   │   │                                #   directement (heatmap de force de couplage par paire de positions, puis
│   │   │                                #   les 4 paires les plus fortes en détail — positions (3,4)/(2,3)/(4,5)/(3,6),
│   │   │                                #   mean|J| hors-diagonale=0.247, max|J|=4.11). Résultats : score GT déterministe
│   │   │                                #   (compute_score(F_potts,J_potts) sur les 68 776 séquences réelles, PAS
│   │   │                                #   held-out — inclut les données d'entraînement de la régression) très corrélé
│   │   │                                #   au target réel (r=+0.889) ; le log enrichment simulé par le Protocol à ce
│   │   │                                #   GT l'est encore un peu plus (r=+0.895 vs réel, r=+0.971 vs le score GT
│   │   │                                #   lui-même) et ne montre AUCUN pic à target1=0 (frac=0.000, contrairement à
│   │   │                                #   la pathologie diagnostiquée avec la GT naïve dans AAV9_fitting_protocol.ipynb
│   │   │                                #   section 4e) — un seul tirage stochastique (pas de moyenne sur plusieurs
│   │   │                                #   répétitions comme la partie 1 d'AAV9_fitting_protocol.ipynb), donc à
│   │   │                                #   confirmer sur plusieurs runs avant d'y voir plus qu'un seul point de mesure.
│   │   │                                # AAV9_potts_GT_fitting_protocol.ipynb (2026-08-27) : rejoue les parties 5
│   │   │                                #   (playground manuel), 6 (overlay score GT vs log enrichment simulé) et 7
│   │   │                                #   (ProfileMLP sur 25 000 variants réels, recovery held-out) d'
│   │   │                                #   AAV9_fitting_protocol.ipynb, même numérotation de sections que les siblings
│   │   │                                #   AAV2/AAV5_fitting_protocol.ipynb (qui rejouent ces mêmes sections pour un
│   │   │                                #   autre dataset AAV, ici c'est le même aav9.csv mais une autre GT) — F_potts/
│   │   │                                #   J_potts au lieu de la GT naïve, hyperparamètres baseline
│   │   │                                #   (mu=50/T_viab=0.8/noise_viab=0.5/D=1e9/RHO_REF=1e-3 dans le code actuel de ce
│   │   │                                #   notebook — PAS encore corrigé vers T_viab=1.3, cf. "État actuel" 2026-08-31 ;
│   │   │                                #   ce notebook a aussi mu=500 dans sa cellule de config, pas 50, incohérence
│   │   │                                #   repérée en marge, ni l'une ni l'autre pas corrigées ici). Version plus complète
│   │   │                                #   qu'AAV9_potts_GT_score_study.ipynb (sections 2/2b/5/6 seulement, pas de
│   │   │                                #   partie 7) : ajoute la partie 7 (ProfileMLP, indépendante du choix de GT,
│   │   │                                #   incluse pour compléter le parallèle avec AAV9_fitting_protocol.ipynb) et une
│   │   │                                #   section 5a nouvelle — tableaux DataFrame de taille de population à chaque
│   │   │                                #   checkpoint du Protocol (lambda0/0p/2/2p/3/3p/4), réutilisant
│   │   │                                #   analysisV1.number_of_seq_threshold/proportion_above_threshold (déjà dans le
│   │   │                                #   projet, pas réinventés) : total, nombre de variants détectés (count>=1), et
│   │   │                                #   nombre de variants au-dessus de seuils de comptage/abondance relative.
│   │   │                                #   Résultat notable : lambda2p/lambda3p (checkpoints NGS post-viability/
│   │   │                                #   sélectivité) ne détectent que 55 713/68 776 (81.0%) des variants réels à
│   │   │                                #   D=1e9 reads — l'essentiel de l'attrition de la librairie vient de la
│   │   │                                #   profondeur de séquençage, pas de la sélection biologique elle-même
│   │   │                                #   (lambda2/lambda3 avant NGS gardent >99.8% des variants détectables). Partie
│   │   │                                #   8 ajoutée après coup : cross-packaging (ProtocolCrossPackagingBackground,
│   │   │                                #   lib/cross_packaging_draft.py, cross_packaging_rate=0.05) comparé au baseline,
│   │   │                                #   même GT/hyperparamètres — r(sim,real) 0.895 (baseline) vs 0.887
│   │   │                                #   (cross-packaging), r(sim,GT) 0.971 vs 0.958. L'hallucination
│   │   │                                #   (ProtocolWithHallucination) a été délibérément exclue de cette section pour
│   │   │                                #   l'instant : le HALLUCINATION_RATE actuellement dans
│   │   │                                #   AAV9_cross_packaging_and_hallucination_impact.ipynb (0.17) ne correspond plus
│   │   │                                #   au taux empirique documenté ailleurs dans le projet (69/74464 ≈ 0.09 %, cf.
│   │   │                                #   new_variant_appearance_analysis.ipynb) et donne un impact largement plus
│   │   │                                #   important (r chute à 0.717 avec ce taux, testé puis retiré) — à corriger/
│   │   │                                #   recalibrer avant de la réintégrer (décision utilisateur 2026-08-27).
│   │   ├── reproducibility/             # (renommé 2026-08-27, corrige la coquille "reproductibility") fit4functionaav9.csv
│   │   │                                #   maintenant publié sur la release GitHub aav-raw-ngs-data-v1 (cf. README) —
│   │   │                                #   gitignoré comme les autres CSV sources, plus le seul CSV source sans mécanisme
│   │   │                                #   de provisioning documenté. log_enrichment_histograms.ipynb (2026-08-26) : lit fit4functionaav9.csv
│   │   │                                #   (gitignoré, 74 464 lignes, colonnes Production1/Production2/Production =
│   │   │                                #   deux réplicats de production + moyenne, ratios de fold-enrichment BRUTS pas
│   │   │                                #   encore log-transformés) ; convertit chaque réplicat en log2 enrichment
│   │   │                                #   (log2(Production1), log2(Production2)) et trace les deux histogrammes
│   │   │                                #   (overlay + côte à côte) pour comparer la reproductibilité entre Production1
│   │   │                                #   et Production2. ~17% des lignes par colonne sont droppées (valeurs 0/inf/NaN,
│   │   │                                #   log2 indéfini) — fraction reportée explicitement plutôt que silencieusement
│   │   │                                #   ignorée.
│   │   │                                # ProfileMLP_train_Production1_test_Production2.ipynb (2026-08-26) : check de
│   │   │                                #   généralisation cross-réplicat -- même ProfileMLP/boucle d'entraînement que
│   │   │                                #   aav_viability_test/aav9/AAV9_profile_model.ipynb, entraîné sur 20% des variants
│   │   │                                #   (log2(Production1) comme cible) et testé sur les 80% restants MAIS avec
│   │   │                                #   log2(Production2) comme cible de test (pas Production1) -- teste si le
│   │   │                                #   signal appris généralise au-delà du bruit de réplicat propre à Production1.
│   │   │                                #   Ne garde que les lignes où Production1 ET Production2 sont valides
│   │   │                                #   simultanément (54 621/74 464, 73.4%) pour que le split 20/80 porte sur les
│   │   │                                #   mêmes variants physiques des deux côtés. `r_replicate_ceiling` (Pearson r
│   │   │                                #   entre log2(Production1) et log2(Production2) sur les lignes de test) sert
│   │   │                                #   de plafond de comparaison : r(MLP, Production2)=0.83 approche mais ne
│   │   │                                #   dépasse pas r(Production1, Production2)=0.89 sur ce même split.
│   │   │                                # ProfileMLP_train40_Production1_test60_Production2.ipynb (2026-08-26) :
│   │   │                                #   même notebook mais split 40% train / 60% test (au lieu de 20/80) — avec
│   │   │                                #   plus de données d'entraînement, r(MLP, Production2)=0.86 se rapproche
│   │   │                                #   encore plus du plafond r(Production1, Production2)=0.89 (quasi identique
│   │   │                                #   au split 20/80, la fraction Production1/Production2 valide simultanément
│   │   │                                #   ne dépend pas du split).
│   │   │                                # new_variant_appearance_analysis.ipynb (2026-08-26) : autre source de bruit
│   │   │                                #   candidate (indépendante du cross-packaging, cf. lib/cross_packaging_draft.py) —
│   │   │                                #   CodonRep1/CodonRep2 sont 2 constructions ADN indépendantes du même variant AA
│   │   │                                #   désigné, mesurées chacune une fois (Production1/Production2). Repère les lignes
│   │   │                                #   où une réplique est à zéro pile (aucun read vecteur détecté) alors que l'autre
│   │   │                                #   est nettement dans le mode "fit" (seuil = vallée entre les 2 modes de
│   │   │                                #   l'histogramme à bins de log2(Production1), find_peaks sur les comptages binnés
│   │   │                                #   — PAS de KDE, cf. consigne permanente utilisateur 2026-08-26 "jamais de KDE,
│   │   │                                #   toujours des bins") : 69/74 464 lignes (0.093%), 100% dépassent le p99 de
│   │   │                                #   désaccord normal entre répliques (mesuré sur les lignes où les deux répliques
│   │   │                                #   sont valides) — donc pas de simple bruit d'échantillonnage. Hypothèse (non
│   │   │                                #   confirmable depuis ce CSV — CodonRep1/2 ne contiennent que la séquence
│   │   │                                #   DESIGNED, pas la séquence réellement observée) : une erreur PCR/synthèse a
│   │   │                                #   changé l'ADN réel d'une seule des 2 constructions, qui cesse silencieusement de
│   │   │                                #   représenter le variant AA désigné. Liste des 69 lignes triées par magnitude
│   │   │                                #   affichée (section 5) — assez peu nombreuses pour être simplement exclues de
│   │   │                                #   l'entraînement en l'état, pas encore de terme dédié dans sequence_classesV1.py.
│   │   │                                #   Section 4 (scatter) annote aussi le nombre de variants sur chacune des 2 lignes
│   │   │                                #   de pile-up au pseudocount (log2(1e-3)≈-9.97, PAS seulement les discordants) :
│   │   │                                #   12 680 sur la ligne Production1=0 (17.0%), 12 805 sur Production2=0 (17.2%),
│   │   │                                #   5 664 sur les deux à la fois.
│   │   ├── mlp_regression/              # expériences de recouvrement MLP (DEEPMLP, ProfileMLP_recovery_nnx, ...)
│   │   │   └── claude_variants/         # variantes exploratoires assistées par IA des mêmes expériences
│   │   └── obsolete/                    # (créé 2026-08-31) notebooks obsolètes, gardés pour référence — pas maintenus,
│   │                                    #   pas dans le flux de travail actif
│   │                                    # AAV_MLP_weights_recovery.ipynb (déplacé depuis aav_viability_test/) : charge
│   │                                    #   sa référence F_viab/J_viab via np.load direct sur lib/{name}_F_viab_mlp.npy/
│   │                                    #   _J_viab_mlp.npy (PAS via initialize_weights.py, jamais basculé vers la GT
│   │                                    #   Potts du 2026-08-27 — compare un MLP fraîchement entraîné à l'ancien
│   │                                    #   model_mlp naïf par construction, pas à une GT de simulation) ; utilisateur
│   │                                    #   prévoit de le compléter/mettre à jour plus tard.
│   │                                    # AAV2_AAV5_error_vs_recovery_check.ipynb (déplacé depuis aav_viability_test/,
│   │                                    #   par l'utilisateur directement) et AAV2_profile_model.ipynb (déplacé depuis
│   │                                    #   aav_viability_test/, sur demande explicite 2026-08-31) — ce dernier vient
│   │                                    #   pourtant d'être basculé vers la régression de Potts jointe (cf. "État
│   │                                    #   actuel" 2026-08-31), donc obsolète pour une autre raison que la GT (pas
│   │                                    #   précisée par l'utilisateur au moment du déplacement).
│   ├── docs/                            # PDF/tex de référence (extraction de poids, encodage one-hot, protocole, ...)
│   └── contrib/                         # export Colab autonome d'un collaborateur, non importé ailleurs
├── Modelization_V2/                     # (créé 2026-08-31) successeur propre, Potts-régression UNIQUEMENT — cf. son
│                                        #   propre README.md pour le détail complet (raison d'être, méthode de
│                                        #   régression avec sources scientifiques, ce qui a été inclus/exclu). Aucune
│                                        #   trace du double-mutant-scan (extract_effective_F/extract_effective_FJ_mlp) ;
│                                        #   autonome (pyproject.toml propre, sys.path.insert vers son propre lib/ —
│                                        #   PAS d'install éditable partagée avec V1, pour éviter toute collision de nom
│                                        #   de module) — vérifié empiriquement (import frais + assertion sur __file__).
│   ├── pyproject.toml                   # nom de package distinct (directed-evolution-modelization-v2), install
│                                        #   éditable optionnelle (pas requise pour faire tourner les notebooks)
│   ├── lib/                             # copies de sequence_classesV1.py/analysisV1.py/RegressionV1.py/
│                                        #   initialize_weights.py/cross_packaging_draft.py (aucune ne contient de
│                                        #   mutant-scan) + aav9_{F,J}_viab_potts.npy (la nouvelle GT) +
│                                        #   aav9_{F,J}_viab_mlp.npy (ancienne GT naïve, gardée UNIQUEMENT parce
│                                        #   qu'AAV9_potts_regression.ipynb s'y compare en interne pour se valider)
│   └── notebooks/                       # AAV9_potts_regression.ipynb (construit la GT), AAV9_potts_GT_score_study.ipynb,
│       │                                #   AAV9_potts_GT_fitting_protocol.ipynb + aav9.csv — les 3 seuls notebooks du
│       │                                #   projet trouvés à la fois propres de mutant-scan ET déjà sur la GT Potts.
│       │                                #   AAV2/AAV5 exclus (aucun de leurs notebooks n'est propre de mutant-scan) ;
│       │                                #   liste complète des fichiers exclus et pourquoi dans le README.md de ce dossier.
│       └── notebooks/Viability/         # (réorganisation manuelle en cours côté utilisateur, d'où le "notebooks/notebooks/"
│                                        #   redoublé — les 3 notebooks ci-dessus y ont été recopiés, plus fit4functionaav9.csv,
│                                        #   AAV9_fit4function_potts_vs_mlp.ipynb, AAV9_potts_simulated_replicate_stochasticity.ipynb,
│                                        #   discordant_variants_aav9.ipynb). Tous ces notebooks localisent lib/ en remontant
│                                        #   jusqu'à Modelization_V2/ (pas de "../../lib" en dur), justement pour survivre à ce
│                                        #   genre de déplacement.
│                                        # AAV9_cross_packaging_parameter_sweeps.ipynb (2026-09-02) : sweeps de paramètres du
│                                        #   protocole avec cross-packaging SEUL (ProtocolCrossPackagingBackground, pas
│                                        #   d'hallucination ni de mutations PCR ; cross_packaging_rate=0 redonne exactement
│                                        #   ProtocolV3, donc le baseline non perturbé est le 1er point du 1er sweep). 5 sweeps
│                                        #   — cross_packaging_rate / mu / T_viab / noise_viab / D — tous autour du même point
│                                        #   de fonctionnement (mu=50, T_viab=1.3 la température propre à la GT Potts,
│                                        #   noise_viab=0.5, D=1e9, rho=1e-3, N0=150*N1, cross_packaging_rate=0.05), un seul
│                                        #   paramètre variant à la fois. Trois figures par sweep : (1) superposition des
│                                        #   histogrammes de log2 enrichment réel aav9 / protocole simulé / prédiction MLP,
│                                        #   un panneau par valeur, en brut ET recalé sur la médiane (le Production réel est un
│                                        #   ratio normalisé, le target simulé un ratio de reads bruts — l'offset log2 constant
│                                        #   entre les deux ne porte pas d'information et n'affecte ni r ni les recoveries) ;
│                                        #   (2) recovery du VRAI fit4function (Pearson r + top-10% + top-500, moyenne des deux
│                                        #   réplicats Production1/Production2, avec le plafond réplicat en référence) pour le
│                                        #   protocole brut ET pour le MLP, en fonction du paramètre balayé ; (3) paysage GT vs
│                                        #   paysage propre du MLP, tous deux sur les 20^7 = 1.28e9 variants (scan exhaustif
│                                        #   chunké sur GPU, ~10 s), avec placement des top-500 que le MLP désigne dans ce même
│                                        #   espace complet (~25 s de scan MLP par point de sweep), plus le top-500 propre à la
│                                        #   GT (le maximum atteignable) et l'étendue GT de la vraie librairie de 74 464 variants.
│                                        #   Le MLP prédit un LOG2 ENRICHMENT — une lecture expérimentale bruitée, dépendante du
│                                        #   tirage aléatoire d'une expérience — alors que la GT produit un SCORE Potts déterministe :
│                                        #   ce ne sont pas deux plages différentes d'une même quantité mais deux objets différents,
│                                        #   donc superposer leurs histogrammes n'a pas de sens et un simple recalage ne le répare pas
│                                        #   (une version intermédiaire ramenait la prédiction brute sur l'axe GT par carte affine —
│                                        #   abandonnée, cf. plus bas). D'où le SURROGATE POTTS (section 4d, conception utilisateur
│                                        #   2026-09-02) : une régression ridge d'un Potts sur les prédictions du MLP, qui transforme
│                                        #   le MLP en objet produisant un SCORE, de la même forme fonctionnelle (F+J) que la GT et
│                                        #   donc directement superposable. Ajusté sur 2 000 000 de variants tirés UNIFORMÉMENT dans
│                                        #   l'espace complet (SAMPLE_IDX), PAS sur les 74 464 de la librairie — point critique : un
│                                        #   surrogate ajusté sur la seule librairie n'est contraint que là, et extrapole précisément
│                                        #   là où vivent les meilleurs picks du MLP (l'argmax sur 1.28e9 est par définition loin de
│                                        #   la librairie designée) ; c'était le défaut de la 1ère version, où les ticks du top-500
│                                        #   tombaient au milieu de la distribution au lieu de son extrême droite. Le surrogate étant
│                                        #   fitté sur du log2 enrichment, son score approxime GT/T_viab : la figure le multiplie par
│                                        #   T_viab (constante connue du protocole, PAS un recalage ajusté) et affiche en regard la
│                                        #   pente empiriquement ajustée comme contrôle de cette relation. Deux astuces pour ne jamais
│                                        #   matérialiser le design 2e6 x 8 541 (68 Go) : X.T@X ne dépend que des séquences et
│                                        #   SAMPLE_IDX est fixe, donc accumulé par chunks et factorisé (Cholesky) UNE fois ; X.T@y
│                                        #   change à chaque point mais toutes les features Potts sont des INDICATRICES, donc ce
│                                        #   produit est juste y sommé par feature — quelques np.bincount, aucune matrice. Chaque fit
│                                        #   se réduit ensuite à une descente triangulaire. Section 11 : TAUX DE RECOUVREMENT TOP-K
│                                        #   entre le classement du MLP (score du surrogate) et celui de la GT, tracé en fonction de k
│                                        #   (de quelques dizaines à des centaines de millions) — la question centrale du notebook en
│                                        #   une courbe. Lu sur une table jointe (score GT x score surrogate) calculée en UNE passe
│                                        #   full-space dans evaluate_point : ses sommes de queue donnent l'intersection à TOUT k,
│                                        #   exact à la résolution des bins, là où un top-k glissant devrait être refait pour chaque k.
│                                        #   Deux conséquences de la grille de bins, documentées dans le notebook : k ne prend que les
│                                        #   valeurs exprimables par les bords de bins GT (espacement irrégulier), et les deux
│                                        #   ensembles top-k ne peuvent pas avoir exactement la même taille, donc le dénominateur est
│                                        #   le plus grand des deux (lecture conservatrice). Assertion de cohérence : la marginale GT
│                                        #   de la table jointe doit égaler l'histogramme GT calculé séparément. Section 11b : recovery
│                                        #   des 500 picks du MLP dans le VRAI top-K global de la GT (500/5k/50k/500k), seuils lus sur
│                                        #   la courbe de survie et recoupés à K=500 contre l'intersection exacte. Section 12 :
│                                        #   RECOMMANDATION DE PROTOCOLE dérivée des données (pas écrite à la main), avec la
│                                        #   distinction explicite entre les deux familles de métriques — r_mlp/prec10/top500 vs
│                                        #   fit4function mesurent le RÉALISME de la simulation (les maximiser reviendrait à régler le
│                                        #   labo pour reproduire l'expérience qu'on a déjà), tandis que gt_recovery_* mesure la capacité à retrouver la
│                                        #   VÉRITÉ SOUS-JACENTE, qui est le but réel du projet et donc le critère optimisé ; les
│                                        #   désaccords entre les deux sont rapportés. Caveats imprimés : un seul paramètre varié à
│                                        #   la fois (la config combinée est une extrapolation non testée), une seule seed par point,
│                                        #   cross_packaging_rate non calibré, baseline dérivée contre l'ANCIENNE GT naïve. Le MLP est
│                                        #   entraîné sur le log enrichment DU PROTOCOLE (jamais sur les colonnes Production
│                                        #   réelles) et évalué sur un split test tenu à l'écart, fixe et partagé par tous les
│                                        #   points de sweep. Tout est en log2 des deux côtés. NON EXÉCUTÉ (demande explicite de
│                                        #   l'utilisateur) — sorties de cellules vides, à lancer avant de faire confiance à un
│                                        #   chiffre ; compter ~30 min sur GPU pour les 30 points de sweep.
└── V0_prototype/                        # prototype première génération, gardé pour l'historique — imports déjà cassés, pas maintenu
```

---

## Description du projet (compris par Claude — à corriger par Aziz)

Objectif : **modéliser mathématiquement le protocole d'évolution dirigée utilisé chez IDV pour les
AAV**, afin de l'optimiser. Le modèle actif est `Modelization_V1/`. On veut trouver quels sont les parametres experimentaux optimaux pour le protocol en laboratoire et construire un outil permettant 
de trouver quels variants sont les plus performants en terme de sélectivité.

**Principe général** : des poids "viabilité" F (additif) et J (épistatique/pairwise) sont extraits
de données AAV réelles (AAV9, notebooks `aav_viability_test/`.) Ces poids servent ensuite à simuler et scorer des populations de variants
(mutants 7-mers, espace combinatoire 20^7 ≈ 1,28e9) pour étudier les régimes de sélectivité
(corrélé / anticorrélé / indépendant entre F et J) et faire tourner la boucle d'évolution dirigée
complète (`directed_evolution_loop/DE_loopV1.ipynb`).
Ensuite via un MLP entraîné sur le profil de
séquence (`ProfileMLP`), on tente de prédire quels variants est le meilleur parmi les 20**7 variants possibles. Actuellement on utilise un MLP car l'expérience etant tres bruitée, on veut exploiter la capacité du MLP a débruité et a capté des interactions complexes entre les acides aminés d'un variant.

**Terminologie à respecter** : la cible réelle et les prédictions du MLP sont des **log enrichment**,
jamais un « score » (ne pas écrire "predicted score", "F_score", "J_score", "total_score", ...).

## État actuel

- **2026-08-31 : `T_viab=1.3` est la température de base pour la GT Potts** (précisé par
  l'utilisateur), **PAS `T_viab=0.8`** — 0.8 reste la valeur dérivée pour l'ANCIENNE GT naïve
  dans `AAV9_fitting_protocol.ipynb` (partie 2, recherche contre l'ancienne GT), jamais
  re-dérivée officiellement pour la GT Potts, mais `AAV9_potts_GT_score_study.ipynb`
  utilisait déjà 1.3 dans son code (confirmé correct). `mu=50`/`noise_viab=0.5`/`D=1e9`
  restent inchangés entre les deux GT. Corrigé dans ce fichier (entrées
  `AAV9_potts_GT_score_study.ipynb`/`AAV9_potts_GT_fitting_protocol.ipynb` ci-dessus) et dans
  `Modelization_V2/notebooks/AAV9/AAV9_potts_GT_fitting_protocol.ipynb` +
  `AAV9_potts_simulated_replicate_stochasticity.ipynb` (ce dernier ré-exécuté avec la bonne
  valeur). **Pas corrigé** : `AAV9_potts_GT_fitting_protocol.ipynb` dans `Modelization_V1`
  (reste à `T_viab=0.8`/`mu=500` dans son code, incohérence pré-existante non résolue — cf.
  entrée détaillée ci-dessus).
- **2026-08-31 : création de `Modelization_V2/`**, successeur propre et autonome ne gardant QUE
  la régression de Potts (aucune trace du double-mutant-scan). Migration sélective depuis V1 —
  quand un fichier était ambigu (notebook pas clairement propre de mutant-scan, ou dataset
  aav2/aav5 sans notebook Potts-only disponible), il a été laissé de côté plutôt que deviné
  (consigne explicite utilisateur : ne pas demander en cas de doute, exclure et rapporter).
  Transféré : `AAV9_potts_regression.ipynb`/`AAV9_potts_GT_score_study.ipynb`/
  `AAV9_potts_GT_fitting_protocol.ipynb` (seuls notebooks du projet trouvés à la fois SANS
  `extract_effective_F`/`extract_effective_FJ_mlp` ET déjà sur les loaders Potts) + leurs
  dépendances lib (`sequence_classesV1.py`/`analysisV1.py`/`RegressionV1.py`/
  `initialize_weights.py`/`cross_packaging_draft.py`, aucune ne contient de mutant-scan) +
  `aav9.csv` + `aav9_{F,J}_viab_potts.npy` + `aav9_{F,J}_viab_mlp.npy` (naïf, gardé seulement
  parce qu'`AAV9_potts_regression.ipynb` s'y compare en interne pour se valider — pas un
  résidu du mutant-scan). Chemins relatifs réécrits pour la nouvelle profondeur (`../lib` au
  lieu de `../../..`), vérifié empiriquement par import frais + assertion `__file__` sous
  `Modelization_V2/` (aucune résolution vers `Modelization_V1/`). PAS d'install pip éditable
  pour V2 dans le venv partagé (collision de nom de module avec V1 sinon) — l'import fonctionne
  via `sys.path.insert` seul, cf. `Modelization_V2/README.md`. Laissé de côté : AAV2/AAV5 (aucun
  de leurs notebooks n'est propre de mutant-scan — `AAV{2,5}_profile_model.ipynb` en ont encore
  dans leur section MLP-recovery malgré la bascule Potts de leur section GT le même jour ;
  `AAV{2,5}_fitting_protocol.ipynb` chargent encore l'ancien npy, pas le Potts) ;
  `AAV9_cross_packaging_and_hallucination_impact.ipynb`, `AAV9_FJ_matrix_top500_check.ipynb`,
  toute la famille `viability_parameter_sweeps/`/`selectivity_weight_regimes/` — hors périmètre
  de cette migration (méthodologie de construction de la GT), pas audités un par un. Détail
  complet dans `Modelization_V2/README.md`, qui documente aussi la méthode de régression de
  Potts elle-même avec sources scientifiques (Weigt et al. 2009 PNAS pour le formalisme
  Potts/champs+couplages ; Otwinowski & Plotkin 2014 PNAS pour l'inférence par régression d'un
  paysage de fitness additif+pairwise et son biais ; Rollins et al. 2019 Nature Genetics pour
  la méthode la plus directement analogue — régression régularisée du même modèle sur des
  données de mutagenèse profonde).
- **2026-08-31 : `AAV2_profile_model.ipynb`/`AAV5_profile_model.ipynb` basculés eux aussi vers
  une régression de Potts jointe (F+J), au lieu du group-means naïf séquentiel — même méthode
  que `AAV9_potts_regression.ipynb`, appliquée cette fois directement DANS les notebooks
  `profile_model` (pas un notebook séparé comme pour aav9). Section 2 ("Building a supposed
  Ground Truth") : `F_groundtruth_viability` remplacée par `RegressionV1.fit_weights_potts_from_data(
  seq_matrix, target, seed=0)` → `F_potts`/`J_potts` calculés ensemble (plus de F d'abord, J en
  résidu ensuite). Section 3b : `J_groundtruth_naive` (group-means + `min_support=5`) supprimée —
  `J_potts` déjà disponible depuis la section 2, comparé au `J_mlp` du scan double-mutant sur tout
  le tableau off-diagonal (plus de `mask_support`/NaN filtering, `J_potts` est dense par
  construction ridge). Nouvelle cellule d'export `lib/aav{2,5}_F_viab_potts.npy`/
  `_J_viab_potts.npy` ajoutée à la suite de l'export MLP existant (même convention que la section 6
  d'`AAV9_potts_regression.ipynb`) — **pas encore de loader dans `initialize_weights.py`** pour
  aav2/aav5 (seul `load_F_viab_aav9_potts`/`load_J_viab_aav9_potts` existent à ce jour). Corrigé au
  passage : les deux notebooks utilisaient `gaussian_kde` (section "1b. Target distribution") pour
  la détection de mode/vallée — remplacé par un histogramme binné (`np.histogram`, 60 bins) +
  `find_peaks` sur les comptages, seule méthode autorisée dans ce projet (cf. consigne permanente
  utilisateur "jamais de KDE" — `new_variant_appearance_analysis.ipynb`/
  `log_enrichment_histograms.ipynb` suivaient déjà cette convention, ces deux-là ne l'avaient pas
  reçue). `AAV9_profile_model.ipynb` n'est PAS touché par ce changement (reste volontairement la
  version naïve/historique, `AAV9_potts_regression.ipynb` étant déjà son pendant Potts en notebook
  séparé). Notebooks non ré-exécutés après ces éditions (sorties de cellules effacées) — à relancer
  avant de faire confiance à un chiffre affiché ; `aav5.csv` (737 587 séquences) rendra la CV ridge
  nettement plus lente que sur aav9 (68 776 séquences), pas encore mesuré.
- **2026-08-31 : notebooks obsolètes déplacés dans `Modelization_V1/notebooks/obsolete/`** —
  nouveau dossier, cf. son entrée dans "Structure du projet" ci-dessus.
- **2026-08-27 : la GT Potts devient la GT par défaut du projet + réorganisation de `notebooks/`.**
  Suite à la validation des résultats d'`AAV9_potts_regression.ipynb` (r prédictif hors-échantillon
  0.847 vs 0.782 pour la GT naïve, cf. entrée `AAV9_potts_regression.ipynb` ci-dessus), l'utilisateur
  a demandé de basculer `F_viab`/`J_viab` vers cette nouvelle GT dans **tous** les notebooks du
  dépôt qui les chargent, de réorganiser `notebooks/` (devenu "le bazar"), et de rendre le dépôt
  clone-and-run pour les CSV sources manquants. Fait :
  1. **Bascule GT** : 22 notebooks (23 candidats trouvés par grep sur `load_F_viab_aav9_mlp`/
     `load_J_viab_aav9_mlp`, moins `AAV9_fitting_protocol.ipynb` — cf. son entrée ci-dessus pour
     pourquoi il reste sur l'ancienne GT) basculés vers `load_F_viab_aav9_potts`/
     `load_J_viab_aav9_potts` par script (remplacement d'identifiant, y compris dans la prose
     markdown qui les cite). **Aucune ré-exécution forcée** (décision utilisateur : "les CSV se
     regénèrent tout seuls") — à la place : sorties de cellules stockées effacées sur les 22
     fichiers (rien de trompeur ne reste affiché à côté d'un code qui charge maintenant une GT
     différente) et tous les caches `diversity*.csv` gitignorés obsolètes supprimés (~130 fichiers,
     keyés par hyperparamètres mais pas par la source de GT — se seraient rechargés
     silencieusement avec les anciennes données sinon). `AAV9_potts_regression.ipynb` (compare les
     deux GT par nom) et `AAV9_profile_model.ipynb` (source de la GT naïve elle-même) gardent
     volontairement les deux loaders / l'ancien loader.
  2. **Réorganisation `notebooks/`** (8 → 6 dossiers, plus aucun nom avec espace) :
     `analysis of correlation/` et `deeper_mlp/` fusionnés (chacun un seul notebook) dans
     `analysis of parameters for viability/`, elle-même renommée `viability_parameter_sweeps/` ;
     `reproductibility/` renommé `reproducibility/` (coquille). Fait via `git mv` (historique
     préservé) — le mécanisme d'import réel du projet est l'install éditable
     (`pyproject.toml`/`package-dir=lib`), pas les `sys.path.insert` de chaque notebook (déjà
     silencieusement sans effet dans plusieurs d'entre eux), donc le déplacement ne casse aucun
     import ; les CSV sont lus en chemin relatif au dossier du notebook, donc sûrs tant qu'ils
     bougent avec lui (vérifié cas par cas avant déplacement).
  3. **Provisioning CSV** : `lib/aav9_{F,J}_viab_potts.npy` trackés dans git (même convention que
     les `.npy` naïfs) ; `fit4functionaav9.csv` (seul CSV source sans mécanisme de provisioning —
     ni release, ni auto-génération) uploadé sur la release GitHub publique existante
     `aav-raw-ngs-data-v1` (`gh release upload`/`edit`) ; `README.md` mis à jour (section "Data &
     derived artifacts" + arbre "Repository structure", qui avait aussi dérivé de la réalité
     indépendamment de ce changement).
  **Non fait délibérément** : ré-exécution des 22 notebooks basculés (laissée à l'utilisateur,
  cf. point 1) ; re-dérivation de mu/T_viab/noise_viab/D contre la nouvelle GT dans
  `AAV9_fitting_protocol.ipynb` (décision utilisateur explicite, cf. son entrée ci-dessus) ; mise
  à jour de la prose markdown citant des chiffres précis calculés sous l'ancienne GT dans les
  notebooks de `selectivity_weight_regimes/` (flaggé dans leur entrée, texte non corrigé — seul le
  code l'est).

- **Fix 2026-08-26 : `jax_enable_x64` activé globalement dans `sequence_classesV1.py`.** Bug
  repéré en creusant un pic suspect (au lieu d'un dégradé) sur le mode "viable" d'un histogramme
  `target1` simulé dans `AAV9_fitting_protocol.ipynb` (cellule manuelle, `F_viab**2`/`J_viab**2`) :
  `jax.random.poisson()` retourne un `int32` par défaut, qui se fait **silencieusement clamper**
  (pas d'erreur) à `2**31-1` pour tout `rate` au-delà — vérifié sur les 68 776 séquences réelles
  d'aav9 avec cette config exacte, **57.5%** des séquences (39 544/68 776) avaient leur `lambda2`
  écrasé sur exactement la même valeur clampée, détruisant le signal de fitness relatif pour plus
  de la moitié de la librairie. Present aussi (plus discrètement) sur la config de base du
  projet : 138/68 776 séquences déjà clampées avant le fix (invisible en histogramme densité à
  cette fraction, mais bien réel). `jax.config.update("jax_enable_x64", True)` en tout début de
  fichier (doit précéder toute opération JAX) fait passer `jax.random.poisson()` en `int64` par
  défaut (plafond ~9.2e18 au lieu de ~2.1e9) — vérifié : 0 séquence clampée sur la config de base
  du projet après fix (1000/1000 valeurs uniques parmi les 1000 plus hautes, contre déjà des
  doublons avant). `Protocol.compute_score()` cast maintenant aussi `F`/`J` en `float64` en
  interne (les `.npy` de poids réels sont sauvegardés en `float32`, et JAX ne remonte pas
  automatiquement un tableau float32 existant vers float64 même avec x64 activé globalement —
  sans ce 2e cast, `exp(score/T_viab)` pouvait toujours déborder en `inf` en float32, et
  `jax.random.poisson(inf)` renvoie silencieusement `0` au lieu d'un grand nombre). **Résidu
  connu, non corrigé** : le cas `F_viab**2`/`J_viab**2` de `AAV9_fitting_protocol.ipynb` reste
  si extrême (rate jusqu'à ~1e48 pour la séquence la plus haute) qu'il dépasse aussi ce que
  `jax.random.poisson()` peut échantillonner correctement même en float64/int64 (limite propre à
  l'algorithme interne de JAX à cette échelle, pas un problème de dtype) — 3 649 séquences
  clampées au nouveau plafond int64 et 6 toujours à `lambda2=0` sur cette config précise
  (c'était déjà 6 avant le fix). Alternative écartée (proposée mais non retenue) : clipper le
  score avant `exp()` + repli sur une approximation Normale(rate, sqrt(rate)) au-delà du plafond,
  qui aurait évité toute activation globale de x64 (coût : ~2x mémoire sur tous les tableaux
  float/int JAX du projet, tous notebooks confondus) — gardée en tête si le résidu ci-dessus
  devient gênant. **Tous les Protocol/ProtocolV2/ProtocolV3/ProtocolBacterialCFU + les 2 classes
  de `lib/cross_packaging_draft.py`** re-testés après le fix (`N_loop_DE` sur 1-2 rounds,
  sorties toutes finies) — aucune régression détectée.
- **Convention depuis 2026-08-21 : pool d'évaluation fixe cross-notebook.** Tous les notebooks de
  `analysis of parameters for viability/` et `deeper_mlp/diversity_sweep_deeper_mlp.ipynb`
  incluent maintenant une section "Fixed 50,000-sequence evaluation pool" : `EVAL_POOL_KEY_SEED=999`,
  `EVAL_POOL_SIZE=50_000` (mêmes valeurs partout — réutiliser exactement ce couple pour rester
  comparable). Corrige le problème du split train/test interne au sweep qui devient dégénéré à
  petit pool (ex. `d0=200` → test fold de 100 séquences → `topk_recovery(k=1000)` se clampe
  trivialement à 100%) — dans les notebooks où `d0` varie (`diversity_sweep.ipynb`,
  `diversity_sweep_adaptive_D.ipynb`, `deeper_mlp/diversity_sweep_deeper_mlp.ipynb`), la recovery
  top-K en fonction du test fold interne au sweep a été **retirée** (gardée seulement pour
  Pearson r, qui ne dégénère pas de la même façon) au profit du pool fixe. En plus du score GT
  (déterministe) et de la prédiction MLP (inférence pure) sur ce pool fixe, chaque notebook simule
  aussi un `protocol_eval` **séparé** (jamais mélangé au pool d'entraînement, ce qui fausserait
  `mu`/`D` à petit `d0` — cf. discussion du 2026-08-21) pour obtenir un vrai `GT<->protocole` sur
  ce pool commun : simulation unique si `mu`/`rho`/`D`/`T_viab`/`noise_viab` sont tous fixes
  (`diversity_sweep.ipynb`), re-simulée à chaque point si le paramètre balayé affecte
  `protocol_eval` aussi (`mu_HEK_multiplicity_sweep.ipynb` sections 7.6/7.8/8.1/8.3 où `mu` est
  balayé, `T_viab_sweep.ipynb`, `noise_viab_sweep.ipynb`, `D_sequencing_depth_sweep.ipynb`,
  `diversity_sweep_adaptive_D.ipynb` où `D` dépend de `d0`). `mu_HEK_multiplicity_sweep.ipynb`
  (le dernier à recevoir cette convention) ajoute ses sections 7.8/8.3 SANS renuméroter le reste
  (insérées juste avant les sections `## 8.`/à la toute fin, cf. son propre historique de
  croissance par ajout de sections plutôt que d'insertion).
- **Convention depuis 2026-08-24 : `mu=10` + `N0=150*N1` dans la famille "diversity" (`d0` swept
  sur une large plage).** Contrainte labo : `N1` (cellules HEK transfectées) doit rester au moins
  150x plus petit que `N0` (copies de plasmide dans la prep), donc `N0=150*N1` remplace les
  anciennes conventions incohérentes (`N0=N1*10` dans certaines cellules, `N0=1e9` fixe dans
  d'autres, parfois les deux dans le même notebook). Combiné à `rho=1e-4`, ce ratio plafonne `N0`
  à un ordre de grandeur raisonnable (~5e12) seulement si `mu` reste modéré — d'où le passage de
  l'ancien défaut `mu=50` à `mu=10` pour cette famille, et la grille `DIVERSITY_GRID` plafonnée à
  `d0=200 000` (au lieu de `1 000 000`) dans `diversity_sweep.ipynb` et
  `diversity_sweep_adaptive_D.ipynb`. `deeper_mlp/diversity_sweep_deeper_mlp.ipynb` avait déjà sa
  propre grille `200`-`200 000` (jamais poussée à `1M`), donc seul son `mu`/`N0` a changé, pas sa
  grille. **Ne s'applique qu'à cette famille** (`d0` variant sur une large plage) — `mu_HEK_
  multiplicity_sweep.ipynb` (où `mu` est justement la variable balayée), `T_viab_sweep.ipynb`,
  `noise_viab_sweep.ipynb`, `D_sequencing_depth_sweep.ipynb` gardent `mu=50` à `d0=20 000` fixe,
  ce qui ne pose pas ce problème (`N0` y reste largement sous le plafond même à l'ancien ratio).
  **Écart non résolu** : `deeper_mlp/diversity_sweep_deeper_mlp.ipynb` utilise `noise_viab=3` (pas
  `0.5` comme `diversity_sweep.ipynb`) — repéré en marge de ce changement, pas corrigé (flag
  ouvert, décision à prendre par Aziz). Les 3 notebooks concernés ont leurs sorties de cellule
  effacées (paramètres changés, anciens résultats plus valides) — à ré-exécuter.
- Travail récent concentré sur `aav_viability_test/` et `selectivity_weight_regimes/` : clarification
  que les notebooks "brute-force top-K global" (sur les 20^7 séquences théoriques) et "top-500 réel"
  (sur la vraie librairie AAV9, held-out test split) **ne sont pas contradictoires** — ils scorent des
  populations différentes du même paysage F/J appris (le premier montre J dominant, le second F
  dominant : normal, pas un bug).
- 12 des 15 notebooks concernés annotés d'encadrés bleus de désambiguïsation (type de population :
  brute-force théorique / librairie NGS réelle / baseline synthétique aléatoire / split test held-out)
  — nécessitent une couleur de texte explicite, sinon invisibles en mode sombre Jupyter.
- **2 bugs identifiés mais pas corrigés** en marge de ce travail :
  1. `MLP_bilinear_head_anticorrelated.ipynb` (section "With a good dataset") ré-entraîne sur le pool
     aléatoire d'origine (`X_train_full`) au lieu de la librairie designed à 427 050 séquences qu'il
     prétend utiliser (MSE de validation identique confirme le bug).
  2. `MLP_for_correlated_weights.ipynb` et `MLP_for_independent_weights.ipynb` affichent un label
     figé "200 000"/"200k" alors que le pool réel est de 2 000 000 ou 20 000 000 de séquences selon le
     notebook.
  3. **(repéré 2026-08-26, casse `initialize_random_weights()`)** `build_J()` (`sequence_classesV1.py`,
     modifié dans le commit "pre sequences classes update") valide maintenant que `interactions.shape
     == (7, 7, 20, 20, 1)`, mais `initialize_random_weights()` lui passe toujours une simple liste
     Python de tuples `(i, j, a, b, value)` — plante avec `AttributeError: 'list' object has no
     attribute 'shape'` dès qu'on appelle `initialize_random_weights()`. Casse potentiellement tout
     notebook qui en dépend pour générer des poids aléatoires ; pas corrigé (semble être un edit en
     cours, pas terminé).
- **Convention depuis 2026-08-21** : tous les nouveaux constructeurs `Protocol`/`ProtocolV2`/`ProtocolV3`
  passent `multinomialNGS=True` (reads NGS via `Multinomial(D, proportions)` au lieu de la Negative
  Binomial surdispersée) — appliqué à tous les notebooks existants. `dataset_filename()` (cache CSV de
  `build_or_load_dataset`, dans les notebooks de `selectivity_weight_regimes/` et `analysis of
  correlation/`) inclut maintenant ce type de NGS dans le nom de fichier (`ngs_part`), pour éviter
  qu'un changement de `multinomialNGS` ne recharge silencieusement un vieux CSV généré sous l'autre
  régime.
- **Données & `.gitignore`** : les CSV `aav{2,5,9}.csv` (données NGS brutes, `aav_viability_test/`)
  restent gitignorés (trop volumineux, `aav5.csv` ~90 Mo) mais sont publiés en asset sur la release
  GitHub `aav-raw-ngs-data-v1` (lien + instructions dans `README.md`). Les `.npy` dérivés
  (`lib/aav{2,5,9}_{F,J}_viab_mlp.npy`, ~250 Ko au total) sont désormais **trackés dans git**
  (ne sont plus gitignorés) pour que le dépôt soit exécutable dès un clone frais sans regénération.



---

*Corrige ce fichier librement (description, état, priorités) — il reflète ma compréhension du projet,
pas une vérité figée.*
