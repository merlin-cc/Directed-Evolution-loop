# CLAUDE.md — Directed-Evolution-loop

> Instructions pour Claude Code sur ce dépôt. Voir aussi `README.md` (setup, régénération des
> données, "Next tasks") — ce fichier-ci est un résumé d'orientation rapide + suivi de structure.

## ⚠ Consigne permanente

**À chaque création d'un nouveau fichier ou dossier significatif dans ce projet (nouveau notebook,
module `lib/`, sous-dossier), mettre à jour la section "Structure du projet" ci-dessous dans le même
tour.** Les données dérivées/gitignorées (`*.csv` de diversité, `*.npy`, `__pycache__`,
`.ipynb_checkpoints`, `.venv`) ne nécessitent pas d'entrée individuelle.

**Pseudocount des log enrichments calculés par ce projet : toujours `eps=0.5`**
(`log2((num+eps)/(den+eps))` ou `log((num+eps)/(den+eps))` selon la base), jamais `+1` ni un
autre epsilon. Convention adoptée le 2026-09-11, après diagnostic sur la viabilité AAV5 : la
colonne `log2_enrichissement_virus_sur_plasmide` fournie dans les CSV IDV ne reconstruit pas
exactement avec un pseudocount `+1` (écart max ~1.4 sur `AAV5_organoides_sorted.csv`), ce qui
la rend incohérente avec le pseudocount `+0.5` déjà utilisé ailleurs dans le projet pour la
pondération inverse-variance (`w = 1/(1/(n_num+0.5) + 1/(n_den+0.5))`,
`AAV5_SEL_potts_regression.ipynb`) et pour `RegressionV1.build_multi_round_dataset`/
`recover_weights_from_NGS` (déjà `eps=0.5` par défaut). S'applique à tout endroit du code qui
**calcule lui-même** un log enrichment à partir de comptages bruts — pas aux colonnes déjà
fournies telles quelles dans les CSV sources (qu'on ne peut pas recalculer avec certitude sans
connaître leur normalisation d'origine). Fichier corrigé pour s'y conformer (2026-09-11) :
`Modelization_V2/notebooks/notebooks/Selectivity/AAV5/AAV5_SEL_fitting_protocol.ipynb`
(`EPS = 1.0` → `0.5`, seul endroit du projet trouvé à dévier de cette convention).

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
│   ├── docs/                            # (créé 2026-09-07) GITIGNORÉ (Modelization_V2/docs/) — notes techniques
│                                        #   locales (PDF + .tex source), contenu dérivé du README.md de V2 +
│                                        #   AAV9_potts_regression.ipynb, rien de neuf côté code :
│                                        #   - GT_construction_V2.pdf : comment la GT V2 est construite — régression ridge
│                                        #     jointe (F+J) d'un modèle Potts contre le log enrichment réel d'aav9.csv, PAS
│                                        #     de la pseudo-vraisemblance/plmDCA ; distingue les deux familles (Weigt 2009 vs
│                                        #     Otwinowski&Plotkin 2014 / Rollins 2019).
│                                        #   - GT_MLE_vs_pseudolikelihood.pdf : en quoi le fit ridge EST un MLE (moindres
│                                        #     carrés = MLE gaussien, ridge = MAP/MLE pénalisé) et pourquoi la
│                                        #     pseudo-vraisemblance n'est pas applicable (elle estime P(s) sur un ensemble de
│                                        #     séquences et ignore les étiquettes ; ici données = paires (séquence, log
│                                        #     enrichment) d'une librairie designée).
│                                        #   - selectivity_aav2_aav5_session_brief.md (2026-09-07) : brief à coller dans une
│                                        #     session fraîche pour traiter les données AAV2/AAV5 organoïdes (cf. ci-dessous).
│   ├── notebooks/notebooks/AAVs dataset/  # ⚠ RENOMMÉ (2026-09-15, geste utilisateur dans l'IDE, en
│                                        #   parallèle de cette session — PAS un `git mv` de Claude, déplacement
│                                        #   filesystem brut) : ancien `notebooks/notebooks/Selectivity/` (AAV2/AAV5,
│                                        #   décrit juste en dessous) fusionné avec l'ancien `notebooks/notebooks/
│                                        #   Viability/` + son duplicata `notebooks/notebooks/notebooks/Viability/`
│                                        #   (AAV9, décrit plus bas dans cette section) en UN SEUL arbre par sérotype :
│                                        #   `AAVs dataset/AAV2/`, `AAVs dataset/AAV5/`, `AAVs dataset/AAV9/`. AAV2 et
│                                        #   AAV5 gardent la structure `viability/`/`selectivity/` ×
│                                        #   `sorting/`/`analysis of noise/`/`analysis of recovery/` mise en place plus
│                                        #   tôt le même jour (cf. réorg ci-dessous) ; AAV9 (viabilité uniquement) est
│                                        #   maintenant sous `AAV9/viability/` (à plat, tous les notebooks AAV9
│                                        #   directement dedans — pas de sous-catégorie sorting/noise/recovery, comme
│                                        #   avant). Toutes les entrées historiques de cette section (ci-dessous et plus
│                                        #   bas, y compris celles sous l'ancien `└── notebooks/`/`└── notebooks/
│                                        #   Viability/`) gardent leur chemin d'écriture d'origine — lire cette note
│                                        #   pour l'emplacement réel actuel. Nouveau : `AAV2/viability/
│                                        #   AAV2_potts_regression.ipynb` (2026-09-15, Claude) — cf. son entrée dédiée
│                                        #   plus bas dans "État actuel".
│                                        # (2026-09-07, ajout utilisateur) AAV2_organoides.csv (4.27M lignes, 34 col) +
│                                        #   AAV5_organoides.csv (5.60M lignes, 35 col) — données NGS réelles IDV
│                                        #   CONFIDENTIELLES, gitignorées (*.csv + entrées nommées explicites), NE JAMAIS
│                                        #   committer/pousser/uploader. Viabilité : compte_plasmide/compte_virus +
│                                        #   log2_enrichissement_virus_sur_plasmide (analogue du target aav9). Sélectivité :
│                                        #   comptes organoïde/noyaux ADN&ARN + log2_enrichissement_*_{adn,arn}_sur_virus
│                                        #   (3 réplicats organoïde, 2 noyaux, + moyenne). Contrairement à aav9, comptes
│                                        #   bruts présents → permet enfin le GLM Poisson count-aware (offset log(plasmide)).
│                                        #   Plan complet dans docs/selectivity_aav2_aav5_session_brief.md. Sous-dossiers
│                                        #   AAV5/ (AAV5_organoides.csv + notebooks), AAV2/ (AAV2_organoides.csv +
│                                        #   notebooks, premier notebook `AAV2_viab_sorting.ipynb` le 2026-09-14, cf.
│                                        #   son entrée dédiée plus bas — travail focalisé viabilité uniquement, pas
│                                        #   de notebook sélectivité AAV2 pour l'instant).
│                                        #   ⚠ RÉORGANISATION (2026-09-11, décision utilisateur) : AAV5/ ne garde comme
│                                        #   notebook ACTIF que `AAV5_SEL_potts_readout_depth.ipynb` — c'est celui qui
│                                        #   donne la meilleure corrélation trouvée sur AAV5 cette session (sel_org2
│                                        #   r=+0.64 à T=50 readout-depth, +0.53 à T=20 ; toujours une régression de
│                                        #   Potts, juste sur un jeu de données filtré par profondeur de lecture plutôt
│                                        #   que par comptage plasmide). **Tous les autres notebooks AAV5/AAV5_SEL_*.ipynb
│                                        #   déplacés (`git mv`) vers `AAV5/obsolete/`** : AAV5_SEL_analysis.ipynb,
│                                        #   AAV5_SEL_potts_regression.ipynb, AAV5_SEL_sorting.ipynb,
│                                        #   AAV5_SEL_fitting_protocol.ipynb, AAV5_SEL_viab_potts_regression_eps05.ipynb,
│                                        #   AAV5_SEL_profile_model_before_after_sorting.ipynb,
│                                        #   AAV5_SEL_deep_profile_model_sel_org2.ipynb (ce dernier créé le même jour,
│                                        #   jamais exécuté — l'utilisateur a repris la main sur l'exécution avant de
│                                        #   demander ce rangement). Les CSV (`AAV5_organoides*.csv`) restent en place
│                                        #   dans `AAV5/`, pas déplacés (pas des notebooks) ; les entrées détaillées
│                                        #   ci-dessous pour chacun de ces fichiers gardent leur chemin historique
│                                        #   `AAV5/AAV5_SEL_*.ipynb` tel qu'écrit au moment des faits — lire
│                                        #   `AAV5/obsolete/AAV5_SEL_*.ipynb` pour les retrouver sur disque aujourd'hui.
│                                        #   ⚠ RÉORGANISATION (2026-09-15, demande utilisateur) : les notebooks ACTIFS
│                                        #   d'AAV2/ et AAV5/ sont déplacés (`git mv`, historique préservé) dans des
│                                        #   sous-dossiers `viability/`/`selectivity/`, eux-mêmes subdivisés par
│                                        #   contenu en `sorting/`, `analysis of noise/`, `analysis of recovery/`.
│                                        #   Nouveau layout : `AAV2/viability/sorting/AAV2_viab_sorting.ipynb` ;
│                                        #   `AAV2/viability/analysis of noise/{AAV2_viab_noise_ceiling,
│                                        #   AAV2_viab_profile_model_denoising, AAV2_viab_profile_model_denoising_v2,
│                                        #   AAV2_viab_profile_model_aux_depth_analysis}.ipynb` (la dernière — technique
│                                        #   de débruitage par feature auxiliaire — rejoint ses notebooks source plutôt
│                                        #   que la catégorie recovery) ; `AAV2/viability/analysis of recovery/
│                                        #   {AAV2_viab_profile_model, AAV2_viab_fitting_protocol,
│                                        #   AAV2_viab_top10k_potts_protocol_mlp, AAV2_viab_top50k_potts_protocol_mlp}.ipynb` ;
│                                        #   `AAV5/viability/sorting/AAV5_viab_sorting.ipynb` ;
│                                        #   `AAV5/viability/analysis of noise/{AAV5_viab_noise_ceiling,
│                                        #   AAV5_viab_noise_ceiling_sans_borne_plasmide}.ipynb` ;
│                                        #   `AAV5/viability/analysis of recovery/AAV5_viab_fitting_protocol.ipynb` ;
│                                        #   `AAV5/selectivity/AAV5_SEL_analysis.ipynb` (reste à la racine de
│                                        #   `selectivity/` — exploration générale viab+sel, ne rentre dans aucune des 3
│                                        #   sous-catégories) ; `AAV5/selectivity/sorting/{AAV5_sel_sorting,
│                                        #   AAV5_SEL_potts_readout_depth}.ipynb` ; `AAV5/selectivity/analysis of
│                                        #   recovery/{AAV5_SEL_fitting_protocol_org2org3,
│                                        #   AAV5_SEL_profile_model_sel_sorting}.ipynb`. `AAV5/obsolete/` **non touché**
│                                        #   (reste à plat, déjà hors du flux actif). **Les CSV (bruts et dérivés)
│                                        #   restent en place** à la racine d'AAV2/ et AAV5/, PAS déplacés — ce sont des
│                                        #   données partagées entre plusieurs catégories (ex. `AAV5_organoides_sorted.csv`
│                                        #   consommé à la fois par des notebooks viab et sel), et chaque notebook les
│                                        #   résout déjà via un fallback location-independent (`Path("X.csv")` sinon
│                                        #   `root / "notebooks/notebooks/Selectivity/AAV{2,5}/X.csv"`, `root` retrouvé
│                                        #   en remontant jusqu'au dossier `Modelization_V2` — idem pour `lib/` via
│                                        #   `root / "lib"`) qui ne dépend pas de l'emplacement du notebook lui-même —
│                                        #   vérifié programmatiquement après coup (tous les chemins CSV/npy référencés
│                                        #   se résolvent encore). **Zéro ligne de code modifiée dans les notebooks
│                                        #   déplacés.** Comme pour la réorg du 2026-09-11 : les entrées historiques
│                                        #   ci-dessous gardent leur chemin d'écriture d'origine (`AAV5/AAV5_SEL_*.ipynb`,
│                                        #   `AAV2/AAV2_viab_*.ipynb` sans sous-dossier) — lire le nouveau layout
│                                        #   ci-dessus pour les retrouver sur disque aujourd'hui. Quelques citations en
│                                        #   prose à l'intérieur de certains notebooks (ex. "cf. `obsolete/AAV5_SEL_
│                                        #   sorting.ipynb`") pointent vers un chemin désormais inexact depuis le nouvel
│                                        #   emplacement du notebook citant — non corrigées (texte non fonctionnel, la
│                                        #   résolution de fichier réelle du code n'en dépend jamais).
│                                        #   ⚠ RÉORGANISATION (2026-09-18, demande utilisateur) : nouveau dossier
│                                        #   `AAV2/obsolete_dense_matrix_potts_regression/` — archive (nom explicite :
│                                        #   c'est la MÉTHODE DE RÉGRESSION qui a changé, pas juste un rangement) des 6
│                                        #   notebooks AAV2 dont le rôle est de FITTER F/J par régression de Potts (pas
│                                        #   les notebooks downstream qui consomment déjà des `.npy` exportés) :
│                                        #   `AAV2_viab_sorting.ipynb`, `AAV2_potts_regression.ipynb`,
│                                        #   `AAV2_viab_top{10,50}k_potts_protocol_mlp.ipynb`,
│                                        #   `AAV2_SEL_potts_readout_depth.ipynb`,
│                                        #   `AAV2_SEL_potts_proportional_agreement.ipynb` — chacun reconstruit à son
│                                        #   emplacement actif d'origine avec le nouveau solveur matrix-free par défaut
│                                        #   du projet (`RegressionV1.fit_weights_potts_from_data_matrixfree`, cf.
│                                        #   entrée dédiée dans "État actuel" 2026-09-18 et `Modelization_V2/README.md`
│                                        #   section "Update 2026-09-18"). Reconstruction MÉCANIQUE (même signature
│                                        #   d'appel/retour que l'ancienne fonction) : uniquement
│                                        #   `R.fit_weights_potts_from_data(` → `R.fit_weights_potts_from_data_matrixfree(`
│                                        #   partout + nettoyage des quelques prints `rank={...}/8541` (le nouveau
│                                        #   solveur itératif n'a pas de diagnostic de rang SVD, `rank` vaut toujours
│                                        #   `None`) + note markdown ajoutée en tête de chaque notebook. Structure,
│                                        #   sweeps, heatmaps, exports, sections : INCHANGÉS — seul le moteur de fit
│                                        #   change. Sorties de cellules effacées sur les 6 (méthode de fit changée,
│                                        #   anciens chiffres plus valides) — **notebooks préparés mais jamais exécutés
│                                        #   par Claude** (`feedback_user_runs_notebooks`, retour à la convention
│                                        #   habituelle après la dérogation ponctuelle des 2 notebooks MLE/matrix-free
│                                        #   eux-mêmes exécutés plus tôt le même jour). Logique validée par smoke-test
│                                        #   sur données réelles (le code exact d'`AAV2_viab_sorting.ipynb` rejoué sur
│                                        #   un sous-échantillon de 4 000 lignes, CV + lam=0, sans erreur).
│                                        # AAV5/AAV5_SEL_analysis.ipynb : analyse des hyperparamètres (D par checkpoint,
│                                        #   classements de comptage, contamination 7m8) + régression Potts GT viab +
│                                        #   sélectivité (3 réplicats organoïde). Section 3 : `R.fit_weights_potts_from_data`
│                                        #   avec `LAM` configurable — la CV tombait sur des λ énormes (1e3–2e5) qui écrasent
│                                        #   F/J, donc passé à `LAM=0.0` (fit non régularisé lstsq). Export tagué
│                                        #   `aav5_{F,J}_{name}_potts_{unreg|lam<x>|cv}.npy` (ne réécrit plus les fichiers λ-CV).
│                                        #   Section 1b ajoutée (2026-09-11) : charge en plus `AAV5_organoides_sorted.csv`
│                                        #   (CSV trié) pour comparer visuellement brut vs trié — grille 2×3 d'histogrammes
│                                        #   log-log (bins log-espacés sur les valeurs >0, jamais de KDE, fraction de
│                                        #   `count=0` en légende) : plasmide / virus / organoïde ADN (3 réplicats
│                                        #   superposés), une ligne par CSV. Plasmide nettement resserré par le tri (bornes
│                                        #   10-500, coupe la traîne du contaminant 7m8 à ~4.8M) ; virus quasi identique
│                                        #   brut/trié (le tri ne filtre jamais `compte_virus`, cohérent avec le diagnostic
│                                        #   déjà fait ailleurs que ce comptage reste le facteur limitant non traité pour
│                                        #   `viab`) ; organoïde bimodal dans les deux CSV, org1 nettement plus creux
│                                        #   (98% de zéros) que org2/org3. Exécuté (nbconvert).
│                                        # AAV5/AAV5_SEL_potts_regression.ipynb (2026-09-09) : récupération dédiée des poids
│                                        #   de Potts sur le CSV de TRAVAIL (AAV5_organoides_sorted.csv), 4 cibles = viab +
│                                        #   sel_org{1,2,3}, un fit indépendant chacune (fit_weights_potts_from_data,
│                                        #   LAM=0.0 OLS min-norm, poids inverse-variance, split 50/50, N_FIT=80k/N_EVAL=150k).
│                                        #   Visus : diag CV/hexbin held-out, heatmaps F + mean|J_ij| + top paires couplées,
│                                        #   histos score GT sur les 3.82M variants, r(F/J sel vs viab) + accord réplicats +
│                                        #   hexbin score sel vs viab (régime), recouvrement top-k réplicats, résidus.
│                                        #   Résultats (held-out r) : viab +0.204 (faible — le cap plasmide≤500 du CSV de
│                                        #   travail retire la large dynamique qui portait le signal viab, cf. +0.246 sur le
│                                        #   CSV brut dans AAV5_SEL_analysis), sel_org1 +0.142 (n_finite 79k seulement),
│                                        #   sel_org2 +0.399, sel_org3 +0.360 ; org2/org3 s'accordent (r(F)=+0.85, top-50k
│                                        #   ∩ 0.75), org1 est l'outlier (r(F)~0.4) ; sélectivité faiblement ANTIcorrélée à
│                                        #   la viabilité (r(F_sel,F_viab) ≈ -0.25 pour org2/org3). Exporte
│                                        #   aav5_{F,J}_{name}_potts_sorted_unreg.npy (gitignorés ; suffixe sorted_ distinct
│                                        #   des exports de AAV5_SEL_analysis qui fittent sur le CSV brut). Partie 1 exécutée
│                                        #   (nbconvert). PARTIE 2 (§8-11, ajoutée 2026-09-09, NON exécutée — CV lente, ~10-20
│                                        #   min/cible) : rejoue les 4 fits avec CV K-fold sur λ (lam=None, mêmes splits que
│                                        #   la partie 1 via RNG rejoué), diag CV MSE vs λ + hexbin, table comparant λ=0 vs CV
│                                        #   (r held-out, r(F,F_cv)/r(J,J_cv), r(score), top-500 ∩), histos score superposés,
│                                        #   export aav5_{F,J}_{name}_potts_sorted_cv.npy.
│                                        # AAV5/AAV5_SEL_viab_potts_regression_eps05.ipynb (2026-09-11) : teste si le
│                                        #   pseudocount de `log2_enrichissement_virus_sur_plasmide` (colonne fournie dans le
│                                        #   CSV IDV, utilisée telle quelle par AAV5_SEL_potts_regression.ipynb) explique le
│                                        #   plafond bas de viab (r=+0.204). Recalcule la cible NOUS-MÊMES à partir des
│                                        #   comptages bruts avec `eps=0.5` (convention projet adoptée ce jour, cf. "Consigne
│                                        #   permanente" en tête de ce fichier) : `log2((compte_virus+0.5)/(compte_plasmide+
│                                        #   0.5))`, au lieu de la colonne fournie (dont le pseudocount réel ne colle pas à
│                                        #   `+1`, écart max ~1.4). Même recette de fit que AAV5_SEL_potts_regression.ipynb
│                                        #   (fit_weights_potts_from_data, LAM=0, poids inverse-variance eps=0.5, split 50/50,
│                                        #   N_FIT=80k/N_EVAL=150k). RÉSULTAT : r(cible recalculée, cible CSV)=+0.998 (quasi
│                                        #   un simple décalage constant, mean diff=-1.06, std=0.28) ; held-out r=+0.202,
│                                        #   STATISTIQUEMENT IDENTIQUE au +0.204 de la cible CSV fournie — le pseudocount
│                                        #   n'est PAS la cause du plafond bas, qui reste un vrai plafond de signal (cohérent
│                                        #   avec le diagnostic déjà fait en session : filtrer par profondeur compte_virus ne
│                                        #   l'améliore pas non plus, contrairement à sel_org2/sel_org3, cf.
│                                        #   AAV5_SEL_potts_readout_depth.ipynb ci-dessous). Exporte
│                                        #   aav5_{F,J}_viab_eps05_potts_sorted_unreg.npy (gitignorés, tag `_eps05_` distinct
│                                        #   de aav5_{F,J}_viab_potts_sorted_unreg.npy). Exécuté (nbconvert, GPU, <1 min).
│                                        # AAV5/AAV5_SEL_potts_readout_depth.ipynb (2026-09-09) : DIAGNOSTIC du « la CV
│                                        #   n'arrive pas à fitter la sélectivité ». Constat clé : le tri de la §5 filtre
│                                        #   compte_plasmide (axe VIABILITÉ) mais la cible sélectivité = log2(organoïde_adn/
│                                        #   virus) et NI le num NI le dénom n'ont été filtrés (compte organoïde médian 2/19/55
│                                        #   selon réplicat parmi les y finis, compte_virus médian 9). MÉTHODE INCHANGÉE (Potts,
│                                        #   moindres carrés gaussiens, ridge L2, mêmes poids inverse-variance) — seule la
│                                        #   constitution du jeu de fit change : garder compte_organoïde_adn ≥ T ET
│                                        #   compte_virus ≥ T, balayer T ∈ {0,5,10,20,30,50,100}. Cibles sel_org2/sel_org3
│                                        #   (org1 = outlier, écarté). §1 plafond r(y2,y3) vs T (0.70 → 0.88 à ≥10 → 0.93 à
│                                        #   ≥30). §2 sweep λ=0 (EXÉCUTÉ par l'utilisateur) : r_self org2 0.39→0.53→0.64
│                                        #   (T 0→20→50), r(F2,F3) 0.87→0.96, r(J2,J3) 0.60→0.78 — le filtrage marche. §3 CV
│                                        #   grille λ ÉLARGIE np.logspace(-4,8,21), T∈{0,20,50}, vérifie si λ* devient
│                                        #   intérieur. §4 heatmaps F/J T=0 vs T=20. §5 coût en diversité. §6 fit MUTUALISÉ
│                                        #   org2+org3 à T_CHOSEN=20 (λ=0 + CV), export lib/aav5_{F,J}_sel_pool_potts_sorted_
│                                        #   readoutT20_unreg.npy (gitignoré). §7 SCATTER score prédit vs log2 enrichment réel
│                                        #   held-out (hexbin + moy(y) par bin ±σ + Pearson/Spearman), 2×2 pool/individuel.
│                                        #   §3/§6 pas encore ré-exécutés après passage à T=20 + grille élargie. §6 (T=20)
│                                        #   a bien tourné une fois → aav5_{F,J}_sel_pool_potts_sorted_readoutT20_unreg.npy
│                                        #   existent (plus les _readoutT30_ d'une version antérieure).
│                                        # AAV5/AAV5_SEL_fitting_protocol.ipynb (2026-09-09) : pendant AAV5 d'
│                                        #   AAV9_potts_GT_fitting_protocol.ipynb — branche les poids Potts récupérés sur
│                                        #   sequence_classesV1.ProtocolV3 et simule UN round loop_DE(), compare le simulé aux
│                                        #   vraies mesures. Corresp. : lambda0p↔compte_plasmide, lambda2p↔compte_virus
│                                        #   (produce_capsids/F_viab), lambda3p↔compte_organoide_i_adn (selectivity/F_sel).
│                                        #   Poids : viab = aav5_{F,J}_viab_potts_sorted_unreg.npy (r≈0.20, faible), sél =
│                                        #   aav5_{F,J}_sel_pool_potts_sorted_readoutT20_unreg.npy (pool org2+org3 T≥20).
│                                        #   Sous-éch. D0_SIM=400k (loop_DE sur 3.82M trop lourd), priorité aux variants avec
│                                        #   vraie mesure sél + fond aléatoire. §1 score GT déterministe vs réel (plafond),
│                                        #   §2 config ProtocolV3 (mu=50/T_viab=T_sel=1/ln2/noise=0.5/D=1e8 — NON calibrés,
│                                        #   repris d'AAV9 ; T=1/ln2 = seul choix motivé, auto-cohérence 2^score) + 3 rounds
│                                        #   (clés diff.) pour 3 réplicats sél simulés, §3 dist viab réel vs protocole
│                                        #   (histos recalés médiane + hexbin), §4 dist sél org2/org3 réel vs protocole +
│                                        #   scatter + accord réplicat-réplicat simulé vs réel, §5 RECOVERY meilleurs variants
│                                        #   (percentile precision_at_k : vrai top-k% retrouvé par score GT vs log enr simulé,
│                                        #   viab + org2 + org3), §6 population par checkpoint. Smoke-test OK (sim_viab std
│                                        #   1.37 / sim_sel std 1.80 vs réel ~2.5/2.9 → distributions simulées plus étroites,
│                                        #   attendu vu poids faibles + params non calibrés). Piste ouverte : recherche
│                                        #   mu/T/noise/D contre AAV5. **Corrigé + ré-exécuté (2026-09-11)** : `EPS = 1.0` →
│                                        #   `0.5` (seul endroit du projet trouvé à dévier de la nouvelle convention
│                                        #   eps=0.5, cf. "Consigne permanente" en tête de ce fichier — n'affecte que
│                                        #   `sim_viab`/`sim_sel`, calculés depuis les reads NGS SIMULÉS, pas les colonnes
│                                        #   réelles chargées du CSV). Exécuté (nbconvert, GPU) : r(score GT viab, réel)=
│                                        #   +0.196, r(score GT sél org2/org3, réel, T≥20)=+0.560/+0.481, accord réplicat
│                                        #   simulé rep0-vs-rep1/rep2=+0.39 (vs réel org2-vs-org3=+0.85 — le protocole
│                                        #   simulé reste moins reproductible que l'expérience réelle à ces paramètres non
│                                        #   calibrés). Chiffres proches de ceux déjà notés ci-dessus (poids F_viab/J_viab
│                                        #   inchangés par ce fix).
│                                        # AAV5/AAV5_SEL_sorting.ipynb (2026-09-09) : notebook de tri/nettoyage du dataset
│                                        #   AAV5. Charge AAV5_organoides.csv, calcule la distance de Hamming de chaque
│                                        #   variant au 7m8 (LGETTRP) une fois. (1) classements comptage-vs-rang log-log par
│                                        #   colonne (plasmide/virus/organoïde adn) avec 7m8+mutants-1 marqués — au top du
│                                        #   plasmide, effondrés après viabilité. (2) histogrammes des log2 enrichments bruts
│                                        #   (viab + 3 sélectivité), fraction finie par panneau (bins, jamais KDE). (3) table
│                                        #   des variants hamming≤2 du 7m8 triés par comptage plasmide (274 variants : 66 à
│                                        #   1 mut, 207 à 2) + top-30 plasmide annoté `lié_7m8` (révèle d'AUTRES contaminants
│                                        #   -- NB §1 écrit aussi AAV5_organoides_classements.csv (~380 Mo, gitignoré) :
│                                        #   1 ligne/variant, rang (1 = plus abondant, ex-aequo method="min") dans plasmide/
│                                        #   virus/organoïde adn 1-2-3, trié par rang plasmide --
│                                        #   à hamming 5-7, ex. TPANSTK/IADNRVS, virus élevé + log enr ≈8 — vrais binders ou
│                                        #   autres spike-ins, PAS des misreads 7m8) ; masque `contam_mask` proposé (hamming≤2
│                                        #   ET plasmide≥5e3, + 7m8 lui-même = 4 variants) NON sauvegardé. (4) comptage des
│                                        #   variants à faible profondeur : viab<5 (plasmide ET virus) = 710 613 (12.7%),
│                                        #   sel<5 (organoïde 1&2&3) = 5.13M (91.7% — les comptes ADN organoïde sont
│                                        #   quasi tous <5), les deux stricts = 703 743 (12.6%). (5) CSV DE TRAVAIL :
│                                        #   retire hamming≤2 du 7m8 (274) ∪ comptage plasmide hors [10, 500] (1 773 022)
│                                        #   → 1 773 143 retirés, 3 822 400 conservés (68.3%) ; histogrammes log enr brut
│                                        #   vs trié ; écrit AAV5_organoides_sorted.csv (~389 Mo, gitignoré
│                                        #   **/*organoides*.csv) — LE csv de travail du projet AAV5 sélectivité (bornes
│                                        #   plasmide fixées après les explorations §6/§7).
│                                        #   (6) balayage du seuil bas de comptage plasmide (20→100 par 10, 7m8 hamming≤2
│                                        #   toujours retiré) : diversité conservée chute vite — plasmide≥20 → 2 240 457,
│                                        #   ≥50 → 343 369, ≥100 → 20 029 (brut 5 595 543) ; retirer en plus plasmide>500
│                                        #   ne coûte que ~87 variants (seulement 105 variants au-dessus de 500 en tout).
│                                        #   Par seuil : distributions log2 enr (overlay + grille 9×4, backdrop brut) +
│                                        #   table stats — quand le seuil plasmide monte, viab_med chute (-0.59 → -2.09 de
│                                        #   20 à 100) et sel_frac_fini monte (0.13 → 0.23). (7) symétrique : borne inf
│                                        #   fixée à 10, borne SUP décroissante (500→20) sur la fenêtre 10≤plasmide≤sup ;
│                                        #   perte quasi nulle jusqu'à sup=150 (3.82M), puis 3.20M à sup=40, 1.65M à sup=20
│                                        #   (beaucoup de variants ont un comptage plasmide 10-20) ; viab_med REMONTE
│                                        #   (-0.15 → +0.55) en retirant les variants très abondants (qui se dépletent).
│                                        #   §6 et §7 ne réécrivent PAS le CSV (exploration seule). Exécuté (nbconvert).
│                                        #   §8 ajoutée (2026-09-11) : 2e CSV de travail, critère INDÉPENDANT de la borne
│                                        #   plasmide de la §5 — garde les variants avec au moins 1 count sur au moins un
│                                        #   des 3 réplicats organoïde ADN (union `compte_organoide_{1,2,3}_adn > 0`), même
│                                        #   retrait 7m8 (hamming≤2) que la §5 mais PAS de borne plasmide (les deux tris
│                                        #   restent isolés, pas empilés) — objectif : isoler l'effet de "au moins une
│                                        #   mesure organoïde" pour un MLP entraîné en aval
│                                        #   (AAV5_SEL_profile_model_before_after_sorting.ipynb). 587 931/5 595 543 lignes
│                                        #   conservées (10.5% — dominé par org2, le réplicat le moins creux, cf.
│                                        #   AAV5_SEL_analysis.ipynb §1b : 91.3% de zéros pour org2, 98.1%/95.8% pour
│                                        #   org1/org3). Écrit AAV5_organoides_sorted_organoide.csv (gitignoré,
│                                        #   **/*organoides*.csv). Exécuté (nbconvert).
│                                        # AAV5/AAV5_SEL_profile_model_before_after_sorting.ipynb (2026-09-11, étendu à 3
│                                        #   CSV le même jour) : ProfileMLP (même archi que AAV9_profile_model.ipynb,
│                                        #   one-hot 140 -> MLP -> 1 scalaire, MSE NON pondérée) entraîné sur les 4 cibles de
│                                        #   log enrichment (viab + sel_org1/2/3, mêmes colonnes que
│                                        #   AAV5_SEL_potts_regression.ipynb), sur **3 CSV** — brut (AAV5_organoides.csv),
│                                        #   trié (AAV5_organoides_sorted.csv, §5 d'AAV5_SEL_sorting.ipynb, filtre
│                                        #   plasmide), trié_organoïde (AAV5_organoides_sorted_organoide.csv, §8 du même
│                                        #   notebook, filtre "≥1 count sur ≥1 des 3 réplicats organoïde", indépendant du
│                                        #   filtre plasmide) — **12 modèles au total**, split 50/50 par cible puis
│                                        #   sous-échantillonnage à N_FIT=N_EVAL=150 000 (moins pour sel_org1/2/3, dont le
│                                        #   nombre de lignes finies est bien plus faible que viab). §5 grille prédit vs
│                                        #   réel (hexbin + Pearson r), §6 grille recovery top-10% (plot_topk_recovery), §7
│                                        #   tableau delta (chaque CSV trié moins brut) sur r et top-1/5/10/20%. Résultats
│                                        #   (held-out r, brut / trié / trié_organoïde) : viab +0.286 / +0.276 / **+0.316**
│                                        #   (meilleur résultat des trois, alors que n_finite chute à 554 392 contre 5.0M
│                                        #   pour brut — effet de sélection : les variants qui atteignent une mesure
│                                        #   organoïde ont aussi une mesure de viabilité de meilleure qualité, pas juste un
│                                        #   gain de volume) ; sel_org1 +0.195 / +0.192 / +0.195, sel_org2 +0.407 / +0.438 /
│                                        #   +0.405, sel_org3 +0.345 / +0.366 / +0.334 — trié_organoïde n'apporte
│                                        #   quasiment RIEN aux 3 cibles sel (n_finite quasi inchangé vs brut : 108 326 vs
│                                        #   108 370 pour org1, etc.) car `isfinite(target)` sur `log2(organoïde_i/virus)`
│                                        #   filtre déjà implicitement "≥1 count sur CE réplicat précis" — le filtre union
│                                        #   sur les 3 réplicats n'ajoute rien pour une cible sel donnée (il aurait fallu
│                                        #   filtrer par réplicat spécifique, pas en union, pour espérer un effet sur sel).
│                                        #   Exécuté (nbconvert, GPU, ~10-12 min pour les 12 entraînements).
│                                        #   ⚠ MISE À JOUR (2026-09-14) : `AAV5_SEL_analysis.ipynb` réactivé (sorti
│                                        #   d'`obsolete/`, geste de l'utilisateur hors de cette session — raison non
│                                        #   documentée) ; `AAV5_SEL_deep_profile_model_sel_org2.ipynb` **supprimé** du
│                                        #   dossier actif par l'utilisateur après comparaison jugée non concluante (Deep
│                                        #   moins bon que Shallow sur `sel_org2`, cf. son entrée ci-dessous — une copie
│                                        #   non exécutée de 2026-09-11 subsiste dans `obsolete/`) ; `AAV5_organoides_
│                                        #   classements.csv` (379 Mo, cache de classement de `AAV5_SEL_sorting.ipynb` §1)
│                                        #   et `AAV5_organoides_sorted_organoide.csv` (75 Mo, filtre "≥1 count organoïde"
│                                        #   de sa §8) supprimés — plus référencés par aucun notebook actif (seuls
│                                        #   `AAV5/obsolete/AAV5_SEL_sorting.ipynb` et
│                                        #   `AAV5/obsolete/AAV5_SEL_profile_model_before_after_sorting.ipynb`, déjà figés,
│                                        #   les citent encore). Deux nouveaux notebooks actifs, cf. entrées dédiées :
│                                        #   `AAV5_sel_sorting.ipynb` (nouveau filtre "sel", pendant du filtre "viab" de
│                                        #   `AAV5_SEL_sorting.ipynb`) et `AAV5_SEL_profile_model_sel_sorting.ipynb`
│                                        #   (comparaison ShallowProfileMLP brut vs les 3 variantes de ce filtre).
│                                        # AAV5/AAV5_SEL_deep_profile_model_sel_org2.ipynb (2026-09-11, exécuté et
│                                        #   supprimé du dossier actif le 2026-09-14 — copie non exécutée conservée dans
│                                        #   `obsolete/`) : compare `ShallowProfileMLP` (~27k params, archi standard) et
│                                        #   `DeepProfileMLP` (reprise verbatim de `Modelization_V1/notebooks/
│                                        #   viability_parameter_sweeps/diversity_sweep_deeper_mlp.ipynb`, ~90k params,
│                                        #   embedding par position + tête pairwise + tête dense 4 couches) sur `sel_org2`,
│                                        #   brut vs `AAV5_organoides_sorted.csv` (filtre `viab`), pleine donnée (pas de
│                                        #   cap `N_FIT`/`N_EVAL`, split 50/50). RÉSULTAT (retenu pour mémoire, le fichier
│                                        #   lui-même est supprimé) : Deep systématiquement LÉGÈREMENT PIRE que Shallow —
│                                        #   held-out r brut 0.412 (deep) vs 0.423 (shallow), trié 0.429 vs 0.441 (Δr
│                                        #   ≈ -0.011/-0.012 partout, aussi sur les top-k%) — la capacité supplémentaire
│                                        #   n'aide pas sur cette cible/ce volume de données, d'où l'abandon de la
│                                        #   variante deep pour la suite (tous les notebooks `sel_org2` suivants
│                                        #   n'utilisent que `ShallowProfileMLP`).
│                                        # AAV5/AAV5_sel_sorting.ipynb (2026-09-14) : pendant du filtre `viab`
│                                        #   (`AAV5_SEL_sorting.ipynb` §5, borne `compte_plasmide`) pour la sélectivité —
│                                        #   nommage adopté cette session : filtre existant = **viab**, celui-ci = **sel**.
│                                        #   Section 1-3 : au lieu de `compte_plasmide`, borne `compte_virus` (dénominateur
│                                        #   du ratio `sel_org_i = organoïde_i/virus`, même rôle structurel que
│                                        #   `compte_plasmide` pour `viab`) + retrait 7m8 (hamming≤2, identique à `viab`).
│                                        #   Balayage seuil bas (1 à 500) et seuil haut (500 à 50 000, la traîne de
│                                        #   `compte_virus` monte à ~3e6, même profil de contamination que le plasmide)
│                                        #   pour choisir `VIRUS_MIN=5`/`VIRUS_MAX=5000` (à tâtons, informé par les
│                                        #   percentiles/tables affichées, PAS un seuil "propre" trouvé analytiquement).
│                                        #   Écrit `AAV5_organoides_sorted_sel.csv` (3 092 351 lignes, sel_org2 fini
│                                        #   392 186). Section 4 : empile en plus un seuil sur `compte_organoide_2_adn`
│                                        #   (numérateur, focus `sel_org2` — la cible qui donne systématiquement le
│                                        #   meilleur signal cette session), balayage puis `ORG2_MIN=2` → écrit
│                                        #   `AAV5_organoides_sorted_sel_org2.csv` (365 290 lignes). Section 5 (ajoutée
│                                        #   après un aller-retour sur la pondération inverse-variance, cf. ci-dessous) :
│                                        #   **filtre par intersection de réplicats** au lieu d'un seuil de magnitude —
│                                        #   `compte_organoide_2_adn > 0` ET `compte_organoide_3_adn > 0` (org1 exclu,
│                                        #   déjà identifié comme outlier dans `AAV5_SEL_potts_readout_depth.ipynb`), moins
│                                        #   le 7m8. Motivation utilisateur : un seuil de comptage ne tranche pas "signal
│                                        #   faible réel" vs "sous-échantillonné" — exiger une lecture indépendante sur les
│                                        #   DEUX réplicats est un critère de fiabilité plus direct. Écrit
│                                        #   `AAV5_organoides_sorted_sel_org2org3.csv` (146 589 lignes, 2.6% du brut — org2
│                                        #   seul=486 844, org3 seul=233 974, intersection=146 633 avant retrait 7m8).
│                                        # AAV5/AAV5_SEL_profile_model_sel_org2_invvar.ipynb (2026-09-14, créé, exécuté,
│                                        #   PUIS SUPPRIMÉ par l'utilisateur — résultat négatif gardé en mémoire ici pour
│                                        #   ne pas retenter la même approche naïve) : au lieu de FILTRER par seuil dur,
│                                        #   pondère chaque ligne dans la loss MSE du `ShallowProfileMLP` par sa précision
│                                        #   estimée `w = 1/(1/(n_num+0.5)+1/(n_den+0.5))` (même formule que la
│                                        #   pondération inverse-variance déjà utilisée pour la régression de Potts,
│                                        #   `eps=0.5` — cf. "Consigne permanente"), `n_num=compte_organoide_2_adn`,
│                                        #   `n_den=compte_virus`, normalisée par sa moyenne sur le fit set. RÉSULTAT :
│                                        #   dégrade `r` held-out PARTOUT (brut 0.429→0.413, `sel_org2` filtré dur
│                                        #   0.453→0.446, top-k% aussi en baisse) — diagnostic : la distribution du poids
│                                        #   brut est extrêmement lourde-queue (percentiles [p0=0.5, p50=7.6, p90=54,
│                                        #   p99=617, max=250 634] sur le brut) donc la moyenne de normalisation (41.2) et
│                                        #   la loss sont dominées par une poignée de lignes à comptage énorme
│                                        #   (contamination probable), au détriment de la masse de données normales —
│                                        #   PAS une réfutation du principe (un plafonnement/winsorisation du poids au
│                                        #   p99 aurait pu corriger ça), juste de l'implémentation non bornée testée ici.
│                                        #   Abandonné au profit du filtre par intersection de réplicats (§5 ci-dessus,
│                                        #   nettement supérieur) plutôt que d'être corrigé.
│                                        # AAV5/AAV5_SEL_profile_model_sel_sorting.ipynb (2026-09-14) : `ShallowProfileMLP`
│                                        #   SEUL (pas de variante deep, cf. `AAV5_SEL_deep_profile_model_sel_org2.ipynb`
│                                        #   ci-dessus) entraîné sur `sel_org2`, pleine donnée (pas de cap `N_FIT`/
│                                        #   `N_EVAL`, split 50/50 seed=0 partagé), **4 CSV** : brut, `sel` (virus seul),
│                                        #   `sel_org2` (virus+org2), `sel_org2∩org3` (intersection de réplicats, §5 d'
│                                        #   `AAV5_sel_sorting.ipynb`). RÉSULTATS held-out r (n_fit) : brut +0.429
│                                        #   (206 909), sel virus +0.430 (166 680), sel_org2 +0.453 (155 249),
│                                        #   **sel_org2∩org3 +0.596 (62 300)** — MEILLEUR RÉSULTAT DE LA SESSION sur
│                                        #   `sel_org2` toutes méthodes confondues (loin devant le Potts non filtré
│                                        #   +0.399 et le ProfileMLP sur `viab`-trié +0.438), avec un saut net sur tous
│                                        #   les top-k% (top-10% 0.304→0.404, top-20% 0.406→0.500) malgré seulement 62 300
│                                        #   lignes de fit (30% du volume `sel_org2` filtré dur, 4x moins que le brut) —
│                                        #   confirme que le critère "détecté sur 2 réplicats indépendants" filtre le
│                                        #   bruit bien mieux qu'un seuil de magnitude sur un seul comptage, cohérent avec
│                                        #   le fait que l'archi (~27k params) n'est de toute façon pas data-starved à ces
│                                        #   volumes (cf. diagnostics similaires ailleurs dans ce fichier).
│                                        # AAV5/AAV5_SEL_fitting_protocol_org2org3.ipynb (2026-09-14) : fait tourner la
│                                        #   classe mécaniste `ProtocolV3` (pas un fit — poids Potts déjà extraits
│                                        #   réutilisés tels quels : viab = `aav5_{F,J}_viab_potts_sorted_unreg.npy`, sél =
│                                        #   `aav5_{F,J}_sel_pool_potts_sorted_readoutT20_unreg.npy`) sur 50 000 variants
│                                        #   sous-échantillonnés de `AAV5_organoides_sorted_sel_org2org3.csv` (§5 d'
│                                        #   `AAV5_sel_sorting.ipynb`), avec la librairie initiale `lambda0` **construite
│                                        #   à partir de `avg(compte_organoide_2_adn, compte_organoide_3_adn)`** au lieu du
│                                        #   `compte_plasmide` réel — choix délibéré de l'utilisateur, pas une
│                                        #   approximation. Paramètres protocole **classiques, sans sweep** (repris de
│                                        #   `AAV5_SEL_fitting_protocol.ipynb`) : `rho=1e-3`, `mu=50`, `T_viab=T_sel=
│                                        #   1/ln(2)`, `noise_viab=noise_sel=0.5`, `D=1e8`, `dilution_factor=1e5`. **Deux
│                                        #   bugs/corrections numériques rencontrés** (documentés dans le notebook, pas
│                                        #   juste patchés silencieusement) : (1) un variant concentrant 13% de la masse
│                                        #   totale de `avg(org2,org3)` faisait exploser la mémoire de `produce_capsids()`
│                                        #   (matrice `(d0,max_cells)`, OOM ~125 Gio testé et confirmé) → winsorisation
│                                        #   p99 (500/50 000 lignes cappées, independante du facteur d'echelle choisi
│                                        #   ensuite) ; (2) **calibration de `lambda0`** (raffinée deux fois) — laisser
│                                        #   `lambda0` à l'échelle brute des comptages NGS (~1e7 au total) fait s'effondrer
│                                        #   `lambda0p` simulé sur 92 variants/50 000 (`dilution_factor=1e5` est calibré
│                                        #   pour des bibliothèques ~1e11-1e12 molécules) ; un premier fix (rescale des
│                                        #   proportions à `N0=150*N1`) marchait déjà bien (98%→95.7% détectés) mais
│                                        #   choisissait une masse totale arbitraire. Version finale (demandée par
│                                        #   l'utilisateur) : `lambda0(s) = plasmid_count(s) * dilution_factor`, pour que
│                                        #   `sample_sequences()` (Binomial(lambda0, p=1/dilution_factor)) retombe **en
│                                        #   espérance exactement sur `avg(org2,org3)`** après dilution — vérifié
│                                        #   numériquement dans le notebook (médiane `lambda0/dilution_factor` = médiane
│                                        #   `plasmid_count` = 57.0) ; `lambda0p` détecte 98.2% des variants (49 119/50 000).
│                                        #   RÉSULTATS FINAUX (n=50 000, sauf viab n=47 693 lignes finies) : viab — plafond
│                                        #   GT r=+0.255, simulé r=+0.266 (quasi identique : le bruit de protocole n'ajoute
│                                        #   presque rien par-dessus la faiblesse déjà connue des poids viab) ; sél org2 —
│                                        #   plafond GT r=+0.555, simulé r=+0.240 (perd environ la moitié du signal) ;
│                                        #   sél org3 — plafond GT r=+0.481, simulé r=+0.207 (ces deux derniers chiffres
│                                        #   INCHANGÉS par la recalibration de `lambda0` — attendu, `lambda2`/`lambda3` ne
│                                        #   dépendent que des PROPORTIONS de `lambda0`, pas de son échelle absolue).
│                                        #   **Accord réplicat-réplicat simulé (rep0 vs rep1/rep2, +0.22/+0.22) très en
│                                        #   dessous du réel org2-vs-org3 sur ce même sous-ensemble (+0.694)** — à ce
│                                        #   point de fonctionnement non calibré pour AAV5, le protocole simulé est
│                                        #   nettement plus bruité que la vraie expérience. Recovery top-1% en `viab`
│                                        #   quasi au niveau du hasard (sim 0.011 vs hasard 0.01, GT 0.032) — le bruit de
│                                        #   comptage à `D=1e8`/2000 reads-variant
│                                        #   écrase le classement aux percentiles extrêmes. Piste ouverte, non faite ici :
│                                        #   une recherche mu/T/noise/D façon partie 2 d'`AAV9_fitting_protocol.ipynb`,
│                                        #   spécifiquement contre ce sous-ensemble org2∩org3, reste à faire pour
│                                        #   calibrer AAV5 (déjà noté comme piste ouverte dans l'entrée `AAV5_SEL_
│                                        #   fitting_protocol.ipynb` ci-dessus, toujours pas fait).
│                                        # AAV5/AAV5_viab_sorting.ipynb (2026-09-14) : reprend le filtre `viab` depuis
│                                        #   zéro, suite à un diagnostic sur `AAV5_SEL_fitting_protocol_org2org3.ipynb`
│                                        #   (§5) — le pic simulé au plancher pseudocount sur `sim_viab` N'EST PAS un
│                                        #   artefact (correction utilisateur) : il reproduit une vraie population
│                                        #   non-fit (variants qui disparaissent réellement à l'étape virus), cohérent
│                                        #   avec la bimodalité déjà documentée de `fit4functionaav9.csv`
│                                        #   (`log_enrichment_histograms.ipynb`). Le vrai problème : le `y_viab` réel
│                                        #   comparé dans ce notebook vient du CSV `_sel_org2org3.csv`, conditionné sur
│                                        #   détection organoïde (org2>0 ET org3>0) — un signal organoïde suppose que le
│                                        #   virus a été produit, donc `compte_virus` y est strictement >0 partout
│                                        #   (vérifié : 0/50 000 lignes à zéro) : cette population réelle exclut
│                                        #   structurellement le mode non-fit, ce n'est pas une preuve que le simulé se
│                                        #   trompe. Ce notebook reprend donc le filtre `viab` sur la population
│                                        #   COMPLÈTE (pas conditionnée organoïde), en repartant de zéro plutôt qu'en
│                                        #   retouchant l'ancien filtre `[10, 500]` sur `compte_plasmide`
│                                        #   (`obsolete/AAV5_SEL_sorting.ipynb` §5, jamais comparé à une version moins
│                                        #   agressive). Étapes : retrait 7m8 seul (274/5 595 543, inchangé, comme
│                                        #   partout ailleurs) puis retrait des SEULS comptages nuls — `compte_plasmide
│                                        #   == 0` : 550 498 lignes (9.84%), EXACTEMENT la fraction déjà non-finie de
│                                        #   `log2_enrichissement_virus_sur_plasmide` fourni (ce filtre n'ajoute donc
│                                        #   rien par rapport à un simple `isfinite()`) ; `compte_virus == 0` : 0 ligne
│                                        #   dans tout le dataset brut (5 595 543 lignes) — jamais nul dans ce CSV
│                                        #   (min=0.5, cf. aussi le check sur `AAV5_organoides_sorted.csv`). Résultat :
│                                        #   5 044 820 variants conservés (90.2% du brut) contre 3 822 400 (68.3%) pour
│                                        #   l'ancien filtre `[10,500]`. MAIS la distribution obtenue est PLUS FRAGMENTÉE
│                                        #   (6 pics `find_peaks`, prominence≥5% : -2.73/-2.13/-1.73/-1.13/+0.27/+1.27)
│                                        #   que celle de l'ancien filtre (4 pics mieux séparés en 2 clusters nets :
│                                        #   -3.25/-2.90/-2.21/+0.89). §6-11 (ajoutées dans le même tour, suite à la
│                                        #   suggestion utilisateur "les pics sont dus à des counts trop bas, des
│                                        #   idées ?") : DITHERING — `compte_plasmide`/`compte_virus` sont des multiples
│                                        #   exacts de 0.5 (100% des lignes), et `compte_plasmide < 5` n'a que 10
│                                        #   valeurs possibles dans tout le CSV, d'où les pics (des centaines de
│                                        #   milliers de variants sans rapport partagent le même couple de comptages,
│                                        #   donc le même ratio). Bruit `Uniform(-0.25,+0.25)` ajouté aux comptages
│                                        #   avant recalcul du ratio (`eps=0.5`, visualisation seule — ne remplace rien
│                                        #   en aval, la pondération inverse-variance déjà en place gère l'incertitude
│                                        #   des petits comptages pour la régression) : sur `df_nz` (comptages nuls
│                                        #   seuls retirés), le dithering fait ENTIÈREMENT disparaître la bimodalité —
│                                        #   une seule bosse unimodale (§8, histogramme détaillé imprimé). Sur l'ancien
│                                        #   filtre `[10,500]`, la bimodalité SURVIT au dithering (§9 : 2 pics nets
│                                        #   -3.72/-0.23, vallée franche à -2.45) — pas un artefact là. Sweep §10 de
│                                        #   `PLASMID_MIN` (0/2/5/8/10/15/20/30, dithering à chaque seuil, métrique
│                                        #   "vallée/pics" = profondeur de vallée entre les 2 modes, 1=unimodal,
│                                        #   →0=séparé) : unimodal jusqu'à 5 inclus, bimodalité apparaît à 8 (0.699),
│                                        #   se renforce à 10 (0.630, valeur historique), puis rendements décroissants
│                                        #   (15→0.533/20→0.495/30→0.447 pour une perte de diversité sévère
│                                        #   3.82M→1.21M). **§11 conclusion : `PLASMID_MIN=10` n'était pas arbitraire —
│                                        #   il tombe quasiment exactement sur le seuil où la bimodalité réelle devient
│                                        #   robuste ; en dessous, la population est dominée par du bruit de comptage
│                                        #   qui noie tout signal fit/non-fit, même après correction de la
│                                        #   discrétisation. Recommandation : garder `AAV5_organoides_sorted.csv` comme
│                                        #   CSV `viab` de référence, ne PAS le remplacer.** Écrit
│                                        #   `AAV5_organoides_sorted_viab.csv` (5 044 820 lignes, §5) à titre
│                                        #   EXPLORATOIRE uniquement — ne remplace PAS `AAV5_organoides_sorted.csv`,
│                                        #   toujours utilisé tel quel par `AAV5_SEL_potts_readout_depth.ipynb` et par
│                                        #   les CSV `_sel*` dérivés dans `AAV5_sel_sorting.ipynb`. Exécuté (nbconvert).
│                                        # AAV5/AAV5_viab_fitting_protocol.ipynb (2026-09-14) : `ProtocolV3` sur la
│                                        #   phase de VIABILITÉ seule (`F_sel=J_sel=0`, `selectivity()` tourne quand
│                                        #   même via `loop_DE()` mais son résultat est ignoré), sur 50 000 variants
│                                        #   sous-échantillonnés d'`AAV5_organoides_sorted.csv` (le CSV `viab` validé
│                                        #   dans `AAV5_viab_sorting.ipynb`, PAS un pool synthétique). Différence clé
│                                        #   avec les notebooks `AAV5_SEL_fitting_protocol*.ipynb` précédents :
│                                        #   `lambda0` calibré sur le VRAI `compte_plasmide` de chaque variant (pas un
│                                        #   proxy comme `avg(org2,org3)`) — possible ici uniquement parce que le CSV
│                                        #   `viab` a `compte_plasmide` directement, donc aucune winsorisation requise
│                                        #   (déjà borné `[10,500]` par construction du CSV). **`D=1e6` (demande
│                                        #   explicite utilisateur)**, calibré pour matcher le ratio reads/variant réel
│                                        #   global : total réel ~1.16e8 (plasmide)/~1.76e8 (virus) reads pour
│                                        #   5 595 543 variants (~21-31 reads/variant sur le brut, ~27-39 sur le CSV
│                                        #   trié) → `D/d0=20` à l'échelle réduite de ce notebook (50 000 variants),
│                                        #   même ordre de grandeur — au lieu du `D=1e8` (`D/d0=2000`, ~100x trop dense)
│                                        #   repris d'AAV9 dans les notebooks précédents. Reste des paramètres
│                                        #   "classiques" inchangés : `rho=1e-3`, `mu=50`, `T_viab=1/ln(2)`,
│                                        #   `noise_viab=0.5`, `dilution_factor=1e5`. Sanity check : `r(score GT viab,
│                                        #   réel)=+0.201`, cohérent avec les chiffres déjà connus (+0.196 à +0.266
│                                        #   selon le sous-ensemble, notebooks précédents). **RÉSULTAT INATTENDU :**
│                                        #   `r(sim_viab, réel)` CHUTE à `+0.099` (brut) / `+0.092` (après dithering,
│                                        #   même technique qu'`AAV5_viab_sorting.ipynb` — quantum réel=0.5→jitter
│                                        #   ±0.25, quantum simulé=1 (comptages `Multinomial` entiers)→jitter ±0.5) —
│                                        #   BIEN PLUS BAS que les +0.196/+0.266 obtenus à `D=1e8` dans les notebooks
│                                        #   précédents. Le dithering confirme que ce n'est PAS un artefact de
│                                        #   visualisation : la distribution réelle ditherée reste clairement bimodale
│                                        #   (2 pics nets, -2.28/+0.70, cohérent avec `AAV5_viab_sorting.ipynb`), mais
│                                        #   la distribution SIMULÉE ditherée devient UNIMODALE (1 seul pic, +0.18) —
│                                        #   le bruit de mesure simulé à cette profondeur (comptage NGS `Multinomial`
│                                        #   ET bruit de production `noise_viab=0.5`) noie entièrement le signal
│                                        #   déterministe F+J, alors que le vrai signal biologique reste visible dans
│                                        #   les vraies données à ce même ordre de grandeur de comptage. Accord
│                                        #   réplicat-réplicat simulé (rep0/1/2) seulement +0.35 — bruit de protocole
│                                        #   élevé à ce `D`. 86.7% des variants ont `lambda2p>0` (virus détecté).
│                                        #   Interprétation ouverte (pas tranchée ici) : `D=1e6` matche le ratio
│                                        #   reads/variant MOYEN du dataset BRUT, mais le sous-échantillon utilisé ici
│                                        #   vient du CSV `viab` déjà filtré/enrichi (`compte_plasmide` médian=22.5,
│                                        #   dans la zone "informative" identifiée par le sweep d'`AAV5_viab_sorting.
│                                        #   ipynb`) — matcher la moyenne globale du brut n'implique pas forcément que
│                                        #   le bruit de mesure simulé (Multinomial + `noise_viab`) soit calibré au bon
│                                        #   niveau pour reproduire la précision de CE sous-ensemble précis ; `rho`/
│                                        #   `mu`/`T_viab`/`noise_viab`/`dilution_factor` restent tous non recalibrés
│                                        #   pour AAV5 spécifiquement (repris tels quels de la config "classique"
│                                        #   établie sur d'autres runs).
│                                        #   **MISE À JOUR (même jour) : poids F_viab/J_viab refit directement sur CES
│                                        #   50 000 variants** (au lieu de réutiliser les poids globaux fit sur les
│                                        #   3 822 400 lignes complètes), pour trancher si le faible `r(sim,réel)`
│                                        #   venait d'un décalage poids/échantillon ou du bruit de protocole lui-même —
│                                        #   même recette partout ailleurs (`RegressionV1.fit_weights_potts_from_data`,
│                                        #   poids inverse-variance `eps=0.5`, `lam=0`). Un split interne 25k/25k donne
│                                        #   un r held-out de seulement `+0.152` (rank=4842/8541, design rank-déficient
│                                        #   à cette échelle réduite) — PIRE que le fit global réutilisé jusque-là
│                                        #   (`+0.201` sur ce même sous-échantillon) : un Potts fit à seulement 50k
│                                        #   lignes généralise moins bien que le fit sur 3.8M lignes. Les poids finaux
│                                        #   utilisés dans la suite du notebook sont néanmoins refit sur les 50 000 en
│                                        #   entier (rank=4853/8541) — `r(score GT, réel)` EN ÉCHANTILLON monte à
│                                        #   `+0.333` (partiellement de la sur-adaptation au bruit propre de `y_viab`,
│                                        #   pas seulement du signal biologique — cf. le r held-out plus bas ci-dessus).
│                                        #   r(score local, score global)=+0.411 sur ces 50 000 séquences — paysages
│                                        #   réellement différents, pas juste un bruit de fit négligeable. Avec ce GT
│                                        #   local (le plus favorable possible), `r(sim_viab, réel)` remonte à `+0.179`
│                                        #   (`+0.171` après dithering) — mieux que `+0.099`/`+0.092` avec les poids
│                                        #   globaux, mais **le résultat central ne change pas** : la distribution
│                                        #   simulée ditherée reste UNIMODALE (1 pic, +0.30) alors que la réelle
│                                        #   ditherée reste bimodale (2 pics, -2.28/+0.70, inchangé — ne dépend pas des
│                                        #   poids). Accord réplicat-réplicat simulé quasi identique (+0.36-0.37).
│                                        #   **Conclusion renforcée : le bruit de mesure simulé à `D=1e6` (pas un
│                                        #   décalage de poids GT) est la cause principale de la perte de bimodalité —
│                                        #   même avec le meilleur GT possible pour cet échantillon précis, le
│                                        #   protocole simulé ne reproduit pas la structure fit/non-fit réelle à cette
│                                        #   profondeur.** Piste ouverte (toujours pas faite) : recalibrer `noise_viab`/
│                                        #   `T_viab`/`rho`/`mu` spécifiquement pour AAV5 à ce régime de profondeur.
│                                        # AAV2/AAV2_viab_sorting.ipynb (2026-09-14) : premier notebook AAV2 de
│                                        #   Modelization_V2 — même méthode que la session AAV5 (analyse détaillée,
│                                        #   dithering, sweep informé par les données, régression de Potts), demandée
│                                        #   explicitement par l'utilisateur ("refais pareil"). AAV2_organoides.csv
│                                        #   (4 273 463 lignes) s'avère structurellement DIFFÉRENT d'AAV5 sur plusieurs
│                                        #   points : (1) comptages ENTIERS (quantum=1, pas des multiples de 0.5) —
│                                        #   dithering ajusté en conséquence (±0.5 au lieu de ±0.25) ; (2) `compte_plasmide`
│                                        #   plafonne à 61 seulement (médiane=1, p90=6) contre 500+ pour AAV5 — 34.2% de
│                                        #   lignes à `compte_plasmide==0` (viab non-fini pour 34.2%, cohérent) ; (3)
│                                        #   AUCUN spike-in identifiable comme le 7m8 d'AAV5 côté plasmide (le top-15 par
│                                        #   abondance décroît sans rupture nette, max=61) ; (4) en revanche, contamination
│                                        #   sévère côté VIRUS — `compte_virus` monte jusqu'à 1 081 060 alors que le
│                                        #   plasmide correspondant est ≤9, donnant des `log2 enrichissement` jusqu'à
│                                        #   +19.24 (AAV5/AAV9 ne dépassaient jamais ~+11) ; signature d'un artefact
│                                        #   technique diffus (index-hopping/contamination croisée probable, pas un
│                                        #   spike-in ponctuel — la queue de ratios implausibles est un continuum lisse,
│                                        #   sans rupture nette, donc pas de coupure hamming possible comme pour le 7m8).
│                                        #   Dithering direct sur le brut (population `compte_plasmide>0`) : **UNIMODALE**
│                                        #   dès le départ (1 seul pic, -0.07) — contrairement à AAV5/AAV9. Sweep
│                                        #   `PLASMID_MIN` (0 à 20, dithering à chaque seuil, même métrique
│                                        #   vallée/pics qu'`AAV5_viab_sorting.ipynb`) : reste unimodal jusqu'à la limite
│                                        #   pratique de diversité (n=7258 à `plasmide>20`, déjà proche du plafond de 61) —
│                                        #   **aucune bimodalité fit/non-fit détectée pour AAV2, à aucun seuil testable**,
│                                        #   constat honnête plutôt qu'un échec de méthode (le plafond de comptage
│                                        #   plasmide, 10x plus bas que celui d'AAV5, ne permet simplement pas d'atteindre
│                                        #   le régime où AAV5 révélait sa bimodalité). Filtre `viab` retenu (à
│                                        #   tâtons, comme les filtres AAV5 équivalents) : `PLASMID_MIN=1` (exclut
│                                        #   seulement `compte_plasmide==0`, déjà non-fini de toute façon) + `RATIO_MAX=100`
│                                        #   sur `virus/plasmide` (juste au-dessus du p99=66.9 mesuré sur les lignes à
│                                        #   `plasmide>=10` — cap contamination, PAS motivé par une bimodalité comme pour
│                                        #   AAV5) → 2 746 425 lignes conservées (64.3%). Écrit `AAV2_organoides_sorted.csv`.
│                                        #   Régression de Potts (`RegressionV1.fit_weights_potts_from_data`, poids
│                                        #   inverse-variance `eps=0.5`, split train/test 50/50, `N_FIT=90 000`/
│                                        #   `N_EVAL=120 000` — même recette qu'`AAV5_SEL_analysis.ipynb`) : CV donne un
│                                        #   lambda intérieur à la grille (1e3, PAS au bord — contrairement à AAV5 où la
│                                        #   CV dégénérait vers des lambda énormes), rank=7715/8541. **Held-out
│                                        #   r=+0.266 (CV) vs +0.221 (lam=0 non-régularisé)** — CV retenue, **meilleur
│                                        #   résultat que la viabilité AAV5** (r~0.2) malgré un dataset a priori plus
│                                        #   bruité (comptages entiers, 34% de zéros, contamination virale diffuse).
│                                        #   Exporte `lib/aav2_F_viab_potts_sorted_cv.npy`/`aav2_J_viab_potts_sorted_cv.npy`
│                                        #   (poids du fit sur les 90 000 lignes, PAS un refit sur fit+eval réunis — deux
│                                        #   tentatives de refit sur 210 000 lignes ont fait planter le kernel par manque
│                                        #   de mémoire, le design Potts dense (N,8541) devenant trop volumineux ; corrigé
│                                        #   en exportant directement les poids déjà validés held-out, même convention
│                                        #   qu'`AAV5_SEL_analysis.ipynb` qui n'a jamais fait ce refit non plus). Exécuté
│                                        #   (nbconvert, CPU, ~5 min dont ~4 min de CV). Piste ouverte à l'époque —
│                                        #   adressée le 2026-09-16, cf. `AAV2/selectivity/` plus bas : notebook de
│                                        #   sélectivité AAV2 (organoïde/noyaux).
│                                        # AAV2/AAV2_viab_fitting_protocol.ipynb (2026-09-14) : pendant AAV2
│                                        #   d'`AAV5_viab_fitting_protocol.ipynb` — `ProtocolV3` (viab seule,
│                                        #   `F_sel=J_sel=0`) sur 50 000 variants sous-échantillonnés
│                                        #   d'`AAV2_organoides_sorted.csv`, poids `aav2_F_viab_potts_sorted_cv.npy`/
│                                        #   `aav2_J_viab_potts_sorted_cv.npy`, `lambda0` calibré sur le vrai
│                                        #   `compte_plasmide`, `D=5e5` (`D/d0=10`, encadré par 3.53 plasmide/11.10
│                                        #   virus reads/variant réels sur ce CSV). Inclut un dithering (section 5b,
│                                        #   quantum=1 des deux côtés) pour vérifier que ni le réel ni le simulé ne
│                                        #   sont bimodaux — **écrit avant le feedback utilisateur sur le dithering
│                                        #   ci-dessous ; conservé tel quel, pas retiré rétroactivement, mais plus
│                                        #   aucun nouveau notebook n'en ajoute**. Écrit mais **jamais exécuté** —
│                                        #   l'utilisateur a explicitement demandé à lancer lui-même l'exécution
│                                        #   ("non laisse moi lancer").
│                                        # AAV2/AAV2_viab_profile_model.ipynb (2026-09-14) : `ShallowProfileMLP`
│                                        #   (archi standard du projet, reprise verbatim des notebooks
│                                        #   `AAV5_SEL_profile_model_*.ipynb`) entraîné sur `viab`, brut
│                                        #   (`AAV2_organoides.csv`) vs trié (`AAV2_organoides_sorted.csv`),
│                                        #   `N_CAP=300 000` lignes finies max par CSV (150k fit/150k eval) pour un
│                                        #   temps d'entraînement raisonnable. Écrit mais **jamais exécuté** (même
│                                        #   raison que ci-dessus).
│                                        # AAV2/AAV2_viab_top10k_potts_protocol_mlp.ipynb (2026-09-14) : nouvelle
│                                        #   population de travail demandée explicitement — les **10 000 variants au
│                                        #   plus fort `compte_plasmide`** (représentation la plus fiable de la
│                                        #   librairie initiale), PAS un sous-échantillon aléatoire ni un seuil bas.
│                                        #   Motivation : à ce niveau de comptage (plasmide dans [19,61]), l'effet de
│                                        #   discrétisation qui motivait le dithering ailleurs devient négligeable —
│                                        #   **aucun dithering utilisé ici**, suite au feedback utilisateur ("le
│                                        #   dithering ajoute juste du bruit, je suis pas fan" — cf. mémoire
│                                        #   `feedback_no_dithering`, plus aucun futur notebook n'en utilisera). Même
│                                        #   cap `RATIO_MAX=100` que le filtre `viab` général (57/10 000 lignes
│                                        #   retirées, contamination résiduelle même à ce niveau de comptage) →
│                                        #   9 943 variants retenus, reads/variant réels 23.5 (plasmide)/28.2 (virus)
│                                        #   — bien plus équilibrés que sur la population complète, confirmant que la
│                                        #   contamination expliquait l'essentiel de l'asymétrie observée ailleurs.
│                                        #   Pipeline en 3 étapes SUR LA MÊME POPULATION (split 80/20, pas 50/50, vu
│                                        #   que ~9 900 lignes pour 8541 features Potts laisse peu de marge) : (1)
│                                        #   régression de Potts (CV vs lam=0, export
│                                        #   `aav2_{F,J}_viab_potts_top10k_{cv|unreg}.npy`) ; (2) `ProtocolV3` avec
│                                        #   ces poids, `D=2.5e5` (`D/d0≈25`, calibré sur CETTE population) ; (3)
│                                        #   `ShallowProfileMLP` (même archi que les autres notebooks profile_model) ;
│                                        #   table de comparaison finale des 3 r held-out. Écrit mais **jamais
│                                        #   exécuté** (même raison — l'utilisateur lance lui-même les notebooks
│                                        #   coûteux depuis cette session, cf. mémoire `feedback_user_runs_notebooks`).
│                                        # AAV2/selectivity/sorting/AAV2_SEL_potts_readout_depth.ipynb (2026-09-16,
│                                        #   nouveau dossier `AAV2/selectivity/` — premier notebook de sélectivité
│                                        #   AAV2) : port direct de `AAV5/selectivity/sorting/
│                                        #   AAV5_SEL_potts_readout_depth.ipynb`, demandé explicitement comme "filtre
│                                        #   analogue au meilleur filtre trouvé avec AAV5" — même modèle de Potts
│                                        #   (F+J, 8541 features), même loss/solveur (`fit_weights_potts_from_data`),
│                                        #   mêmes poids inverse-variance `eps=0.5`. Le filtre : `compte_organoide_adn
│                                        #   ≥ T` ET `compte_virus ≥ T` appliqué à la fois à org2 ET org3 (correspondance
│                                        #   entre réplicats + minimum de comptage combinés en un seul critère, PAS
│                                        #   juste `>0`), balayé sur `THRS=[0,5,10,20,30,50,100]`, org1 exclu du fit
│                                        #   (outlier sur AAV5, à reconfirmer sur AAV2 — §1 du notebook). Différence
│                                        #   avec le port AAV5 : `AAV2_organoides_sorted.csv` (produit par
│                                        #   `AAV2_viab_sorting.ipynb` §8) ne garde que les colonnes viab
│                                        #   (`sequence`/`compte_plasmide`/`compte_virus`/cible viab, `usecols`
│                                        #   restreint à l'époque) — pas les colonnes organoïde dont ce notebook a
│                                        #   besoin. Charge donc `AAV2_organoides.csv` BRUT (34 colonnes) directement
│                                        #   et redérive en ligne le même filtre qualité-viab que
│                                        #   `AAV2_viab_sorting.ipynb` §8 (`PLASMID_MIN=1`, `RATIO_MAX=100` sur
│                                        #   `virus/plasmide`) comme population de base, avant d'empiler le seuil de
│                                        #   profondeur du readout par-dessus. Pas de retrait type 7m8 (hamming) :
│                                        #   `AAV2_viab_sorting.ipynb` §3 n'a trouvé aucun spike-in côté plasmide
│                                        #   analogue au 7m8 d'AAV5 pour ce dataset. 9 sections (plafond de
│                                        #   reproductibilité vs T, sweep λ=0, check CV à quelques T, heatmaps F/J
│                                        #   (une ligne par seuil de `THRS`, PAS juste T=0/T_CHOSEN — bug corrigé le
│                                        #   même jour, cf. plus bas), coût en diversité, fit mutualisé org2+org3 à
│                                        #   `T_CHOSEN` + export, scatter score-vs-réel held-out, histogramme du
│                                        #   score GT, conclusion) — structure identique au notebook AAV5 source.
│                                        #   **⚠ Décision (2026-09-16) : `T_CHOSEN=5`, pas 20** (le choix de départ
│                                        #   repris tel quel du port AAV5) — contrairement à AAV5, `r(F2,F3)` (accord
│                                        #   des poids entre réplicats) sur AAV2 n'est PAS monotone croissant en `T` :
│                                        #   il culmine à `T=0-5` (+0.645/+0.647) puis CHUTE jusqu'à `T=30` (+0.185,
│                                        #   `T=20` tombait précisément dans ce creux à +0.259) avant de remonter à
│                                        #   `T=100` (+0.733, mais seulement 6-7k variants). Exporte
│                                        #   `lib/aav2_{F,J}_sel_pool_potts_sorted_readoutT5_unreg.npy` (gitignorés) —
│                                        #   les fichiers `..._readoutT20_unreg.npy` d'un run antérieur à cette
│                                        #   décision restent sur disque mais ne sont plus le choix de production.
│                                        #   Logique validée par un smoke-test sur un échantillon aléatoire ~12% du
│                                        #   CSV brut (pipeline complet : sweep, fit mutualisé, top-k recovery,
│                                        #   scatter) — **notebook lui-même jamais exécuté par Claude dans son
│                                        #   ensemble** (`feedback_user_runs_notebooks`). Bug corrigé le même jour
│                                        #   (édition IDE utilisateur, pas Claude) : la grille de heatmaps §4 avait
│                                        #   `plt.subplots(7, 3, ...)` mais la boucle ne parcourait que `[0, T_CHOSEN]`
│                                        #   (2 valeurs) — 5 lignes sur 7 restaient vides ; boucle corrigée pour
│                                        #   parcourir `THRS` en entier (les poids `W[(T,2)]`/`W[(T,3)]` existaient
│                                        #   déjà pour chaque seuil, calculés en section 2).
│                                        # AAV2/selectivity/analysis of recovery/AAV2_SEL_fitting_protocol_org2org3.ipynb
│                                        #   (2026-09-16) : port direct de `AAV5/selectivity/analysis of recovery/
│                                        #   AAV5_SEL_fitting_protocol_org2org3.ipynb` — `ProtocolV3` (classe
│                                        #   mécaniste, viab+sélectivité) simulée sur 50 000 variants sous-échantillonnés
│                                        #   de la population org2∩org3, pour voir si le protocole simulé reproduit des
│                                        #   distributions/corrélations plausibles. Différence avec le port AAV5 :
│                                        #   aucun CSV org2∩org3 pré-construit n'existe pour AAV2 (contrairement à
│                                        #   `AAV5_organoides_sorted_sel_org2org3.csv`) — section 0 (nouvelle, pas dans
│                                        #   le notebook source) construit cette population EN LIGNE directement depuis
│                                        #   `AAV2_organoides.csv` brut : filtre qualité-viab (`plasmid≥1, ratio≤100`,
│                                        #   même que `AAV2_viab_sorting.ipynb`, pas de retrait 7m8 — aucun trouvé pour
│                                        #   AAV2) puis intersection stricte `compte_organoide_2_adn>0 ET
│                                        #   compte_organoide_3_adn>0`. Poids : sélectivité =
│                                        #   `aav2_F_sel_pool_potts_sorted_readoutT5_unreg.npy` (T=5, décision du jour,
│                                        #   cf. entrée précédente) ; viabilité = `aav2_F_viab_potts_sorted_cv.npy`
│                                        #   (lignée organoïde/haute-diversité d'`AAV2_viab_sorting.ipynb`) — **PAS**
│                                        #   `aav2_F_viab_potts_plasmid50_vectorpos_cv.npy` (la GT canonique par défaut
│                                        #   décidée dans `AAV2_potts_regression.ipynb` §12c), délibérément écarté ici
│                                        #   car il vient d'un CSV totalement différent (basse diversité, `aav2.csv`) —
│                                        #   choix motivé pour rester dans la même lignée de données (mêmes séquences
│                                        #   organoïde) que les poids de sélectivité, comme le fait déjà le notebook
│                                        #   AAV5 source (qui n'a qu'une seule lignée viab, donc where ce choix n'y
│                                        #   était pas ambigu). Paramètres protocole "classiques" NON recalibrés pour
│                                        #   AAV2 — repris tels quels du point de fonctionnement `AAV5_SEL_fitting_
│                                        #   protocol.ipynb` (`rho=1e-3, mu=50, T_viab=T_sel=1/ln(2),
│                                        #   noise_viab=noise_sel=0.5, D=1e8, dilution_factor=1e5`), aucun sweep. Même
│                                        #   winsorisation p99 de `plasmid_count=avg(org2,org3)` (calibration
│                                        #   `lambda0=plasmid_count*dilution_factor`) que le port AAV5, même raison
│                                        #   numérique (`produce_capsids()` OOM sans cap). 9 sections (miroir du
│                                        #   notebook source, section 0 en plus) : sous-échantillon + librairie
│                                        #   initiale, poids Potts, score GT déterministe vs réel (plafond), config
│                                        #   `ProtocolV3` + 3 réplicats simulés, viabilité réel/simulé, sélectivité
│                                        #   réel(org2,org3)/simulé, recovery top-k%, population à travers le pipeline,
│                                        #   notes. Logique validée par un smoke-test sur données réelles (échantillon
│                                        #   25% du CSV brut, poids factices aux bonnes dimensions puisque
│                                        #   `readoutT5` n'existe pas encore tant que la section 6 du notebook
│                                        #   readout-depth n'a pas été relancée à `T_CHOSEN=5`) — pipeline complet
│                                        #   (population org2∩org3, simulation `ProtocolV3` 3 réplicats, recovery,
│                                        #   tableaux population) exécuté sans erreur — **notebook lui-même jamais
│                                        #   exécuté par Claude dans son ensemble** (`feedback_user_runs_notebooks`).
│   ├── lib/                             # copies de sequence_classesV1.py/analysisV1.py/RegressionV1.py/
│                                        #   initialize_weights.py/cross_packaging_draft.py (aucune ne contient de
│                                        #   mutant-scan) + aav9_{F,J}_viab_potts.npy (la nouvelle GT) +
│                                        #   aav9_{F,J}_viab_mlp.npy (ancienne GT naïve, gardée UNIQUEMENT parce
│                                        #   qu'AAV9_potts_regression.ipynb s'y compare en interne pour se valider) +
│                                        #   aav9_{F,J}_viab_{poisson,gamma}.npy (2026-09-07, GT GLM expérimentales, gitignorées,
│                                        #   écrites par AAV9_poisson_regression_GT_score_study.ipynb, PAS de loader).
│                                        #   RegressionV1.py : ajout (2026-09-07) de fit_weights_glm_from_data(seq_matrix,
│                                        #   target_ratio, power) — même design Potts / unpacking F,J que
│                                        #   fit_weights_potts_from_data, mais fit via un GLM Tweedie log-link sur un RATIO
│                                        #   d'enrichissement non-négatif au lieu d'une ridge gaussienne sur le log
│                                        #   enrichment. power=1 → Poisson, power=2 → Gamma. Solver lbfgs (PAS
│                                        #   newton-cholesky : celui-ci forme la hessienne 8540×8540 et, aux petits alpha
│                                        #   que ces données veulent, tourne jusqu'à max_iter → ~1h/fit) ; CV sur un
│                                        #   sous-échantillon de 25k lignes, refit final sur tout ; grille alpha défaut
│                                        #   logspace(-3,2,8) (minimum CV intérieur vers alpha≈5e-3, bien sous les ~24 de
│                                        #   la ridge). ~1-2 min/famille. Caveat docstring : objectif Poisson
│                                        #   scale-équivariant → sans comptages réels (offset), fit dominé par la queue
│                                        #   enrichie ; r(score,target réel) ≈ 0.75 (Poisson) / 0.76 (Gamma) vs 0.89 (ridge).
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
│                                        #   genre de déplacement. NB : AAV9_potts_regression.ipynb/AAV9_potts_GT_score_study.ipynb
│                                        #   (les plus anciens) utilisent ENCORE le "../../lib" en dur qui, depuis ce sous-dossier,
│                                        #   ne pointe nulle part → import silencieux du lib pip-installé de Modelization_V1
│                                        #   (bug pré-existant repéré 2026-09-07, non corrigé).
│                                        # AAV9_poisson_regression_GT_score_study.ipynb (2026-09-07) : rejoue
│                                        #   AAV9_potts_GT_score_study.ipynb (heatmaps J, distribution du score GT sur la
│                                        #   librairie réelle, recovery par percentile, target1 simulé, overlays) mais avec
│                                        #   une GT construite par GLM Tweedie log-link (RegressionV1.fit_weights_glm_from_data)
│                                        #   sur le RATIO exp(target) au lieu de la ridge gaussienne sur le log enrichment :
│                                        #   Poisson (power=1, le sujet) + Gamma (power=2, référence) + ridge Potts (incumbent)
│                                        #   comparés côte à côte. r(score GT, target réel) ≈ 0.75 (Poisson, alpha≈5e-3) /
│                                        #   0.76 (Gamma, alpha≈3e-2) / 0.89 (ridge Potts) — les deux GLM traînent (~-0.13
│                                        #   en r) car l'objectif Poisson scale-équivariant est dominé par la queue enrichie
│                                        #   (pas de comptages → pas d'offset). Écrit aav9_{F,J}_viab_{poisson,gamma}.npy
│                                        #   (gitignorés, pas de loader). ~5-8 min à exécuter (lbfgs, CV sur sous-éch. 25k).
│                                        # AAV9_viab_potts_lam0_vs_cv.ipynb (2026-09-09) : pendant AAV9 d'
│                                        #   AAV5_SEL_potts_regression.ipynb — même procédé λ=0 (OLS min-norm) vs CV sur λ,
│                                        #   ici sur la viabilité aav9.csv (68 776 variants, split 50/50 random_state=0).
│                                        #   Diag CV MSE vs λ + hexbin held-out, heatmaps F/mean|J_ij|, histos score GT sur les
│                                        #   68 776 variants (λ=0 / CV / GT projet superposés), table r held-out + r(F)/r(J
│                                        #   off-diag)/r(score) + recouvrement top-k, résidus. RÉSULTAT (le point de l'exo) :
│                                        #   contrairement à AAV5, λ=0 et CV donnent quasi la même chose — r held-out 0.838
│                                        #   (λ=0) vs 0.847 (CV λ=23.95, minimum INTÉRIEUR propre, pas au bord de grille),
│                                        #   r(F)=1.00 r(J off-diag)=0.988 r(score)=0.991 entre les deux, top-500 ∩ 0.75.
│                                        #   AAV9 est confortablement en régime n>p (68 776 obs, rang 7715/8541) donc l'OLS
│                                        #   est déjà bien conditionné ; la pathologie AAV5 (CV → λ 1e3-2e5 au bord, écrase
│                                        #   F/J) vient de n_eff << p sur des cibles log-ratio très bruitées. GT projet
│                                        #   (fit données complètes) r=0.887 mais optimiste (inclut idx_test). Corrige le
│                                        #   `../../lib` en dur (→ V1 sans arg `lam`) par la remontée vers Modelization_V2/lib.
│                                        #   CV ~14 min (5 folds × ~3 min). Exécuté (nbconvert). Pas d'export de poids.
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
│                                        # test_for_sweep.ipynb (2026-09-07) : LE notebook de sweep de paramètres de référence
│                                        #   (le style de courbes validé "ensemble" avec l'utilisateur). GT Potts, données
│                                        #   fit4functionaav9.csv, ProtocolCrossPackagingBackground. Diffère de
│                                        #   AAV9_cross_packaging_parameter_sweeps.ipynb : PAS de surrogate Potts ni de scan
│                                        #   full-space — un échantillon FIXE de 10M variants tirés de 20^7, l'over-recouvrement
│                                        #   top-k MLP↔GT y est une intersection d'ensembles exacte (argsort), le log enrichment
│                                        #   du MLP n'est jamais converti en score. 3 figures par sweep : plot_distributions
│                                        #   (réel vs protocole vs MLP, un panneau/valeur, recalé médiane), plot_recovery (r /
│                                        #   top-10% / top-1000 vs le param, protocole & MLP, + plafond réplicat + baseline
│                                        #   REAL_MLP_BASELINE=MLP entraîné sur la vraie Production1), plot_topk_recovery
│                                        #   (recouvrement exact top-k MLP↔GT vs k + aux profondeurs GT_RECOVERY_DEPTHS).
│                                        #   7 sweeps : dilution_factor, initial diversity d0 (2026-09-07, sous-échantillonne
│                                        #   la librairie réelle, reconstruit split+protocole+MLP par point, mu tenu donc
│                                        #   N1=mu*d0/rho, D FIXE donc reads/variant monte quand d0 baisse ; fig 2 top-1000 sature
│                                        #   à petit d0 quand le test fold < 1000 → fig 3 est la figure comparable pour ce sweep),
│                                        #   cross_packaging_rate, mu, T_viab, noise_viab, D. Baseline test_for_sweep : D=1e8,
│                                        #   dilution_factor=1e6 (≠ cross_packaging_parameter_sweeps qui a D=1e9). Résumé combiné
│                                        #   exporté → parameter_sweeps_summary.csv (export, jamais rechargé). ~45-55 min GPU.
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

- **2026-09-18 (suite) : plafonds de population `N_FIT`/`N_EVAL` retirés dans les 2 notebooks
  reconstruits qui en avaient un réellement motivé par la mémoire** (`AAV2_viab_sorting.ipynb` :
  `N_FIT,N_EVAL=90_000,120_000` ; `AAV2_SEL_potts_readout_depth.ipynb` : `N_FIT=60_000`,
  répété dans 3 sites — sweep §2, CV §3, `S_pool` mutualisé §6). Suite à la question utilisateur
  "on a changé le type de solveur mais est-ce que tu as bien changé la taille des datasets de
  fitting (vu qu'on a plus de ceiling)" : le rebuild mécanique précédent (entrée ci-dessous)
  avait délibérément laissé structure/tailles inchangées, donc ces deux caps — hérités de
  l'ancien solveur dense/SVD, l'un documenté noir sur blanc comme réponse à un crash mémoire à
  210k lignes — étaient restés actifs malgré le nouveau solveur matrix-free qui n'en a plus
  besoin. **Audit complet des 6 notebooks reconstruits avant de toucher quoi que ce soit** :
  seuls ces deux-là avaient un cap réellement lié à la mémoire ; `AAV2_potts_regression.ipynb`
  (aav2.csv, 53 382 séquences) n'en a jamais eu besoin, `AAV2_viab_top{10,50}k_potts_protocol_
  mlp.ipynb` (`N_TOP=50 000`) est un choix de qualité de données délibéré — pas une contrainte
  de calcul, et `AAV2_SEL_potts_proportional_agreement.ipynb` fit déjà sur la population
  filtrée complète sans sous-échantillonnage superposé — ces 4-là **volontairement pas
  touchés**. `fit_i`/`eval_i` (resp. `tr_fit`, `S_pool`) utilisent désormais directement le
  split train/test complet, sans sous-échantillonnage `RNG.choice`. Logique re-vérifiée par
  smoke-test sur données réelles (même code exact, population non cappée) — **notebooks
  toujours jamais exécutés par Claude** (`feedback_user_runs_notebooks`).
- **2026-09-18 (suite) : deux nouveaux notebooks `{AAV2,AAV5}_dataset_overview.ipynb`** — vue
  d'ensemble brute des CSV `{AAV2,AAV5}_organoides.csv`, à la racine de chaque dossier de
  sérotype (pas dans `viability/`/`selectivity/`, puisqu'ils couvrent les colonnes des deux à la
  fois). Trois sections seulement, sur demande explicite ("just mets y des infos importantes") :
  (1) `df.describe()` (percentiles étendus `.01`/`.05`/`.95`/`.99` en plus des quartiles, vu la
  queue lourde des colonnes de comptage) sur toutes les colonnes numériques ; (2) reads à chaque
  checkpoint (une ligne par colonne `compte_*`) — `total_reads`, `reads/variant` sur tout le
  dataset ET parmi les seuls variants détectés (`count>0`, precisé par l'utilisateur), `fraction>0`
  + table de comptage par seuil via `analysisV1.number_of_seq_threshold` (réutilisé tel quel, déjà
  établi dans `AAV2_potts_regression.ipynb` §1b) ; (3) top 100 séquences par colonne (`nlargest`,
  comptages ET log2 enrichissements), stocké intégralement dans un dict `top100[colonne]`, seul le
  top 10 affiché inline par colonne pour rester lisible. Aucun filtre, aucune régression — pur
  profilage descriptif. Logique smoke-testée sur un sous-échantillon réel des deux CSV (300 000
  lignes de tête) — **notebooks préparés mais jamais exécutés par Claude**
  (`feedback_user_runs_notebooks`). **Section 4 ajoutée dans la foulée** (demande explicite,
  "des scatters plots comparant les org1 2 et 3") : 3 scatters pairwise (hexbin + diagonale y=x +
  Pearson r, même style que le "plafond de reproductibilité" déjà utilisé partout ailleurs dans
  ce projet) entre `log2_enrichissement_organoide_{1,2,3}_adn_sur_virus`, tout le dataset, sans
  filtre. Logique re-vérifiée sur les CSV réels COMPLETS (pas un sous-échantillon) : AAV2
  org1-vs-org2 r=+0.446 (n=84 732), org1-vs-org3 r=+0.422 (n=107 326), org2-vs-org3 r=+0.505
  (n=72 671) ; AAV5 org1-vs-org2 r=+0.418 (n=81 449), org1-vs-org3 r=+0.264 (n=77 899),
  org2-vs-org3 r=+0.694 (n=146 633) — org2/org3 systématiquement plus corrélés qu'avec org1 sur
  les deux sérotypes, cohérent avec le diagnostic déjà établi ailleurs dans ce projet (org1 =
  réplicat outlier, écarté des fits de sélectivité AAV5 ET AAV2 pour cette raison).
- **2026-09-18 (suite) : le solveur matrix-free devient LA méthode de régression de Potts par
  défaut du projet (pas une variante expérimentale) — nouvelle fonction drop-in
  `RegressionV1.fit_weights_potts_from_data_matrixfree` + `score_potts` (scorer centralisé),
  et les 6 notebooks AAV2 qui FITTENT du Potts (pas ceux qui consomment déjà des poids
  exportés) reconstruits avec, anciennes versions archivées.** Suite à la démonstration que le
  solveur matrix-free (entrée précédente) bat la ridge classique à pleine échelle : décision
  utilisateur explicite de l'adopter comme nouvelle base, pas comme une technique "à part"
  qu'on continuerait à appeler "matrix free" dans la prose des notebooks.

  1. **`RegressionV1.score_potts(seq_matrix, F, J, bias=0.0)`** (nouveau) : centralise le calcul
     de score Potts par `gather` (`O(N*L²)`, pas de matrice) que CHAQUE notebook de ce projet
     redéfinissait localement comme `score_FJ` — réutilisé maintenant par la CV du nouveau
     solveur ; les notebooks continuent de définir leur propre `score_FJ` local (non touché,
     hors scope de cette passe), mais tout nouveau code peut importer celui-ci à la place.
  2. **`RegressionV1.fit_weights_potts_from_data_matrixfree`** (nouveau) : remplaçant drop-in de
     `fit_weights_potts_from_data` — MÊME signature d'appel (`seq_matrix, target, sample_weight,
     lambdas_grid, k_folds, seed, verbose, lam`) et MÊME forme de retour (`F_hat, J_hat, rank,
     info` avec `info["lam"]`/`info["cv_mse"]`/`info["lambdas_grid"]`/`info["n_obs"]`), donc un
     site d'appel existant peut basculer en renommant seulement la fonction. `lam=None`
     déclenche une CV K-fold — contrairement à `ridge_cv_mse_potts` (matrice dense par fold),
     celle-ci reste matrix-free de bout en bout : chaque fold refit ET sa MSE de validation
     passent par `fit_weights_potts_ridge_matrixfree`/`score_potts`, jamais de matrice. `rank`
     vaut toujours `None` (pas de diagnostic SVD avec un solveur itératif — documenté comme tel
     dans le docstring, pas une régression silencieuse). Validé par smoke-test contre
     `fit_weights_potts_from_data` sur données réelles (aav2.csv, lam=0 ET CV small-grid) :
     `r(F)=r(J)=1.000000`, même lambda choisi par CV, ~4x plus rapide même à 6 000 lignes.
  3. **Archivage + reconstruction des 6 notebooks AAV2 de fit Potts** (portée décidée avec
     l'utilisateur : SEULEMENT les notebooks qui fittent F/J, pas ceux qui rechargent des poids
     déjà exportés — détail complet dans l'entrée `AAV2/obsolete_dense_matrix_potts_regression/`
     de "Structure du projet" ci-dessous). `git mv` vers le nouveau dossier
     `AAV2/obsolete_dense_matrix_potts_regression/` (nom choisi pour être explicite : c'est la
     méthode de régression qui devient obsolète, pas juste un rangement) pour les 4 fichiers
     déjà trackés, `mv` simple pour les 2 fichiers de `selectivity/` (jamais commités, créés
     cette session). Reconstruction MÉCANIQUE à l'emplacement actif d'origine de chacun : swap
     `R.fit_weights_potts_from_data(` → `R.fit_weights_potts_from_data_matrixfree(` (vérifié
     exhaustif par grep : ces 6 notebooks n'appelaient QUE ce wrapper, jamais les fonctions
     bas-niveau `fit_weights_potts_unregularized`/`fit_weights_potts`/`build_potts_features`
     directement — la substitution est donc sûre partout) + nettoyage des prints
     `rank={var}/8541` devenus trompeurs (`rank` est maintenant toujours `None`) + un `int(rank)`
     retiré (`AAV2_SEL_potts_readout_depth.ipynb`, aurait levé `TypeError` sur `None`) + note
     markdown ajoutée en tête de chaque notebook (pointe vers la version archivée + le notebook
     de validation). **Structure, sections, sweeps, heatmaps, exports : inchangés** — seul le
     moteur de fit change, aucune autre logique retouchée. Sorties de cellules effacées sur les
     6 (méthode de fit changée, anciens chiffres plus valides). **Notebooks préparés mais jamais
     exécutés par Claude** (`feedback_user_runs_notebooks`, décision explicite de l'utilisateur
     pour cette passe — contraste avec les 2 notebooks MLE/matrix-free eux-mêmes, exécutés pour
     de vrai plus tôt le même jour). Logique re-vérifiée par un dernier smoke-test rejouant le
     code EXACT (post-édition) d'`AAV2_viab_sorting.ipynb` sur un sous-échantillon réel de
     4 000 lignes (CV + lam=0), sans erreur.
  4. **`Modelization_V2/README.md`** mis à jour (section "Update 2026-09-18: matrix-free solver
     is now the default fitting method", insérée juste après la description de la procédure de
     fit historique) — explique le calcul mémoire, le principe matrix-free (rétropropagation
     d'un `gather` = `Xᵀr`), la validation, et pointe vers cette entrée de CLAUDE.md pour le
     détail des 6 notebooks reconstruits.

  **Non fait délibérément** (portée confirmée avec l'utilisateur) : les notebooks AAV2
  downstream (ProtocolV3, MLP, analyse de bruit — qui rechargent déjà des `.npy` exportés) NE
  SONT PAS reconstruits dans cette passe, ils reprendront les nouveaux poids la prochaine fois
  qu'ils seront touchés — même convention que la bascule GT Potts du 2026-08-27. AAV5/AAV9 non
  concernés — pas de migration rétroactive hors AAV2 pour l'instant.

- **2026-09-18 (suite) : solveur ridge "matrix-free" (`RegressionV1.fit_weights_potts_ridge_matrixfree`,
  message bumpé 1.6→1.7) — valide, 24-65x plus rapide, et RÉVISE la conclusion de l'entrée
  précédente : la ridge à pleine échelle bat la MLE multinomiale, elle ne fait pas juste la
  rattraper.** Suite à la question "pourquoi la ridge est limitée à 90k, tu peux quantifier ?" :
  mesuré empiriquement que `fit_weights_potts_unregularized`/`fit_weights_potts` (matrice de design
  dense matérialisée) consomment en pic RSS **~5x la taille de `X` elle-même** (14.6 GiB à
  N=90 000, 40.1 GiB à N=250 000, ratio constant — dû au solveur SVD `lstsq`/`np.linalg.solve`, pas
  juste au stockage de `X`), plafonnant à ~700-750k lignes sur cette machine (121 GiB RAM) — pas
  90-150k comme le suggérait l'historique du projet (probablement une machine plus contrainte à
  l'époque). Nouvelle fonction : même objectif ridge (`0.5*Σw(Xθ-y)² + 0.5*λ||θ_sans_biais||²`)
  résolu SANS jamais construire `X` — passe avant par `gather` direct sur `seq_matrix` (même calcul
  que `score_FJ`, `O(N)`, pas de matrice `N×p`), gradient via `jax.grad` (la rétropropagation d'un
  `gather` EST l'opération matrix-free `X^T@r` — même principe que le "surrogate Potts" de
  `AAV9_cross_packaging_parameter_sweeps.ipynb`, ici par autodiff plutôt que bincounts manuels),
  résolu par L-BFGS-B (convexe quadratique → même optimum qu'un solve direct).

  Nouveau notebook `AAVs dataset/AAV2/viability/AAV2_potts_ridge_matrixfree_validation.ipynb`
  (**exécuté par Claude**, résultats réels) :
  1. **Validation à n=90 000 (même échantillon que le fit ridge classique)**, `lam=1.0` ET `lam=0.0`
     (cas non régularisé/minimum-norme, rang-déficient — l'équivalence n'était pas garantie a
     priori) : `r(F_classique,F_matrixfree)` et `r(J_classique,J_matrixfree)` **> 0.999999** dans
     les deux cas, `r` held-out identique à la 4e décimale. **Bonus inattendu : 24-65x plus rapide**
     que le solve classique à cette échelle (1.4-1.5s contre 36.5-91.0s).
  2. **Passage à l'échelle complète (n=4 153 463, tout le CSV)** : `r` held-out = **+0.302** —
     **nouveau meilleur résultat de la session pour AAV2 viabilité**, dépasse nettement la ridge
     classique plafonnée à 90k (+0.202) ET la MLE multinomiale à pleine échelle (+0.205, cf. entrée
     précédente). Le saut ridge@90k→ridge@4.15M (+0.202→+0.302) est bien plus net que celui de la
     MLE aux mêmes deux échelles (+0.133→+0.205) — une fois le plafond mémoire levé, la ridge n'est
     pas juste compétitive avec la MLE, elle la dépasse.
  3. **Vraie comparaison F/J à N ÉGAL, enfin possible** (les deux méthodes ayant maintenant tourné
     sur EXACTEMENT les mêmes 4 153 463 lignes) : `r(F_ridge,F_mle)` passe de 0.938 (N dépareillés,
     entrée précédente) à **0.965**, et surtout `r(J_ridge,J_mle)` de 0.296 à **0.627** — confirme
     que l'essentiel du désaccord J observé précédemment venait de l'écart d'échantillon (90k vs
     4.15M), pas d'une différence fondamentale entre les deux objectifs. Désaccord résiduel réel
     (0.627, pas 1.0) — question ouverte, pas creusée plus loin.

  Exporte `lib/aav2_{F,J}_viab_potts_ridge_matrixfree_full.npy` (fit sur les 4 153 463 lignes
  complètes, `lam=1.0`) — **candidat sérieux pour devenir un nouveau défaut de viabilité AAV2**
  (meilleur `r` held-out de tous les fits AAV2 viab de ce fichier à ce jour), mais pas encore
  promu comme tel — décision à prendre par l'utilisateur, comme pour toutes les décisions de GT
  canonique précédentes de ce projet.

- **2026-09-18 : nouvelle méthode de fit alternative — MLE multinomiale directe sur les comptages
  bruts (Fernandez-de-Cossio-Diaz, Uguzzoni & Pagnani 2021, *MBE* 38(1):318-328,
  doi:10.1093/molbev/msaa204), comparée empiriquement à notre ridge sur AAV2 viabilité.**
  Suite à une question utilisateur ("est-ce que leur MLE peut mieux marcher que notre ridge
  regression ?"), nouvelle fonction `RegressionV1.fit_weights_potts_mle_multinomial` (module V2
  uniquement, message bumpé 1.5→1.6) : au lieu de calculer d'abord `y_s =
  log2((N1_s+eps)/(N0_s+eps))` puis de faire une ridge pondérée sur ce ratio (notre méthode
  actuelle), cette fonction écrit directement la vraisemblance multinomiale du papier
  (approximation "rare binding", leurs éq. 1-3, restreinte au cas T=1 — un seul round, exactement
  notre structure viabilité plasmide→virus, et exactement leur propre cas le plus favorable, le
  jeu Olson et al.) et la maximise par L-BFGS (`scipy.optimize.minimize`, gradient via
  `jax.grad`). Aucune matrice de design dense n'est jamais matérialisée (score calculé par
  gather direct sur F/J, coût `O(N)`) — contrairement à `fit_weights_potts_unregularized`/
  `fit_weights_potts`, plafonnées en pratique à ~90-150k lignes par la mémoire d'une matrice
  dense `(N, 8541)`. Pas de terme de biais (non identifiable : une constante ajoutée à tous les
  scores s'annule exactement dans la normalisation softmax). Pseudocount `eps=0.5` appliqué à
  `N0` avant le log, conformément à la pratique du papier lui-même ("we add a pseudo-count of 1/2
  to all counts ... before carrying out the inference") — confirmation que la convention `eps=0.5`
  du projet est déjà alignée avec leur pratique.

  Nouveau notebook `AAVs dataset/AAV2/viability/AAV2_potts_mle_multinomial.ipynb`
  (**exécuté par Claude, dérogation ponctuelle explicite de l'utilisateur** à la convention
  habituelle `feedback_user_runs_notebooks` — résultats réels, pas un smoke-test) sur
  `AAV2_organoides.csv` (4 273 463 lignes, checkpoint viabilité `compte_plasmide`→`compte_virus`) :

  1. **Comparaison à taille égale (n=90 000, le plafond mémoire de la ridge)** : la ridge
     (`r held-out=+0.202`) bat nettement la MLE multinomiale (`r=+0.133`) — **contraire à
     l'hypothèse initiale**. Explication proposée : le papier compare sa MLE à une "empirical
     selectivity" (leur éq. 4) qui est un `h_s` LIBRE PAR SÉQUENCE, sans aucune structure Potts
     partagée — alors que notre ridge fit déjà un F/J PARTAGÉ avec pondération inverse-variance,
     donc capture déjà une bonne partie de l'avantage "mutualiser l'info entre séquences" que le
     papier attribue à sa méthode.
  2. **MLE à pleine échelle (n=4 153 463, tout le dataset moins le split test)** : `r=+0.205`,
     rattrape tout juste la ridge à 90k lignes — et ~3x plus vite (~29s contre ~92s) malgré 46x
     plus de données, grâce à l'absence de plafond mémoire. C'est là son vrai avantage pratique
     pour ce projet, pas la qualité du fit à volume égal.
     `r(F_ridge90k, F_mle_full)=+0.938` (bon accord additif) mais `r(J_ridge90k,
     J_mle_full)=+0.296` (accord faible sur l'épistasie) — question ouverte, pas démêlée entre
     effet d'échelle (90k vs 4.15M) et effet de méthode.
  3. **Réplication de la décimation du papier (fig. 2a/b)**, sur la MÊME population fixe
     (n=90 000, seuls les reads sont sous-échantillonnés via `Binomial(count,d)`) : le résultat
     QUALITATIF se reproduit — la MLE dégrade environ 2x moins vite EN RELATIF que la ridge
     quand la profondeur chute (coverage 31→0.62 reads/variant) : perte relative in-sample
     -17% (MLE) vs -27% (ridge) ; held-out -22% (MLE) vs -40% (ridge). Mais un effet plus modeste
     que la fig. 2 du papier (où leur "ratio naïf sans structure" s'effondre nettement plus que
     n'importe quelle version structurée) — ici la ridge, déjà structurée, part d'un niveau plus
     haut et le garde sur toute la plage testée ; la MLE est plus STABLE en relatif, pas
     meilleure en absolu à ce volume de données.

  **Conclusion retenue** : pas de remplacement de la ridge comme méthode par défaut du projet à
  ce stade — mais un candidat solide si un futur usage a besoin d'exploiter la pleine échelle
  d'un CSV brut (millions de lignes) ou un régime encore plus sous-échantillonné. Exporte
  `lib/aav2_{F,J}_viab_potts_mle_multinomial_full.npy` (fit MLE 4.15M lignes) et
  `lib/aav2_{F,J}_viab_potts_mle_ridge_baseline_90k.npy` (fit ridge 90k lignes, même split,
  pour comparaison reproductible) — aucun des deux n'est destiné à remplacer
  `aav2_F_viab_potts_plasmid50_vectorpos_cv.npy` (la GT canonique décidée le 2026-09-16, lignée
  de données différente, `aav2.csv` basse-diversité). Résultats numériques bruts dans
  `AAV2/viability/mle_multinomial_results/` (gitignoré comme les autres `.csv`/figures dérivées :
  `baseline_results.json`, `decimation_sweep.csv`, `baseline_comparison.png`,
  `decimation_sweep.png`).

- **2026-09-16 (suite) : correction majeure — le cap `RATIO_MAX=100` (hérité de
  `AAV2_viab_sorting.ipynb`) tronquait toute la queue haute du log2 enrichment réel, retiré de
  `AAV2_SEL_potts_readout_depth.ipynb` et `AAV2_SEL_fitting_protocol_org2org3.ipynb` ; retrait
  aussi du recentrage médiane dans ce dernier (valeurs brutes affichées).** Diagnostic déclenché
  par l'utilisateur : `ratio≤100` équivaut par construction à `log2 enrichment ≤ log2(100)=6.644`
  — vérifié sur `AAV2_organoides.csv` : ce cap rayait **100% des variants à `y>6.64`**, déjà 75%
  de ceux à `y>6`, 46% à `y>5` — pas un filtre de bruit de comptage mais une amputation de la
  queue haute (précisément les variants les plus enrichis). Le mur visible dans un histogramme
  affiché par l'utilisateur (recentré médiane, coupure nette à "+2") a été diagnostiqué comme ce
  même cap (`6.599 (max réel) − 4.458 (médiane de cette population) ≈ +2.14`), pas un bug
  log10/log2 comme d'abord suspecté par l'utilisateur — colonne vérifiée `log2(virus/plasmide)`
  à `r=1.0` près. Les deux notebooks ne gardent plus que `compte_plasmide ≥ 1` comme filtre
  qualité (pas de cap sur le ratio). `AAV2_SEL_fitting_protocol_org2org3.ipynb` affiche
  maintenant les histogrammes réel/simulé en **valeurs brutes, non recentrées** (sur demande
  explicite) — le recentrage médiane n'affectait pas `r` (invariance par translation) mais
  déplaçait la position visible du mur, rendant le diagnostic plus difficile.
  **Conséquence en cascade, pas encore corrigée** : `aav2_F_viab_potts_sorted_cv.npy` (poids de
  viabilité utilisés par `AAV2_SEL_fitting_protocol_org2org3.ipynb`) est lui-même fit sur
  `AAV2_organoides_sorted.csv`, produit par `AAV2_viab_sorting.ipynb` avec l'ANCIEN cap
  `ratio≤100` — ce notebook et les 6 autres notebooks viab qui en dépendent
  (`AAV2_viab_fitting_protocol`, `AAV2_viab_profile_model`, `AAV2_viab_top10k_potts_protocol_mlp`,
  `AAV2_viab_top50k_potts_protocol_mlp`, `AAV2_viab_noise_ceiling`,
  `AAV2_viab_profile_model_denoising`) n'ont PAS été touchés — décision utilisateur en attente sur
  s'il faut les corriger aussi. Toutes les sorties de cellules des deux notebooks corrigés ont été
  effacées (résultats calculés sous l'ancien filtre, plus valides) — **`T_CHOSEN=5` (entrée
  suivante) devra être reconfirmé** une fois la sweep rejouée sans le cap, la population changeant
  substantiellement. Notebooks toujours jamais exécutés par Claude dans leur ensemble
  (`feedback_user_runs_notebooks`).
- **2026-09-16 (suite) : `T_CHOSEN=5` retenu pour la sélectivité AAV2 (pas 20), heatmap F/J §4
  corrigée (bug d'édition IDE), et nouveau notebook `AAV2_SEL_fitting_protocol_org2org3.ipynb` —
  simulation `ProtocolV3` complète (viab+sélectivité) avec ces poids.** (1) Dans
  `AAV2_SEL_potts_readout_depth.ipynb`, `T_CHOSEN` passe de 20 à 5 : `r(F2,F3)` (accord des poids
  entre réplicats) n'est pas monotone croissant en `T` sur AAV2 comme il l'était sur AAV5 — il
  culmine à `T=0-5` (+0.645/+0.647) puis chute jusqu'à `T=30` (+0.185, `T=20` tombait dans ce
  creux à +0.259) avant de remonter à `T=100` (+0.733, mais seulement 6-7k variants). Nouvel
  export : `lib/aav2_{F,J}_sel_pool_potts_sorted_readoutT5_unreg.npy`. (2) Heatmap F/J de la
  section 4 (grille `plt.subplots(7, 3, ...)`, éditée dans l'IDE par l'utilisateur) ne parcourait
  que 2 des 7 lignes (boucle sur `[0, T_CHOSEN]` au lieu de `THRS`) — corrigé pour parcourir tous
  les seuils. (3) Nouveau notebook `AAV2/selectivity/analysis of recovery/
  AAV2_SEL_fitting_protocol_org2org3.ipynb` — port direct de `AAV5_SEL_fitting_protocol_org2org3.ipynb` :
  `ProtocolV3` simulée sur 50 000 variants de la population org2∩org3 (construite en ligne, aucun
  CSV pré-fait pour AAV2 contrairement à AAV5), poids sélectivité = le fit `T=5` ci-dessus, poids
  viabilité = `aav2_F_viab_potts_sorted_cv.npy` (lignée organoïde, délibérément PAS la GT
  canonique basse-diversité de `AAV2_potts_regression.ipynb` §12c — lignées de données
  différentes), paramètres protocole "classiques" empruntés tels quels à AAV5 (aucun point de
  fonctionnement combiné viab+sel propre à AAV2 n'existe encore). Détail complet dans les entrées
  dédiées de "Structure du projet" ci-dessous. Logique des deux notebooks validée par smoke-test
  sur données réelles — **aucun des deux notebooks exécuté par Claude dans son ensemble**
  (`feedback_user_runs_notebooks`).
- **2026-09-16 (suite) : nouveau dossier `Modelization_V2/notebooks/notebooks/AAVs dataset/AAV2/
  selectivity/` — premier notebook de sélectivité AAV2, `AAV2_SEL_potts_readout_depth.ipynb`.**
  Port direct de `AAV5/selectivity/sorting/AAV5_SEL_potts_readout_depth.ipynb` sur demande
  explicite ("le filtre qu'on utilisera sera analogue au meilleur filtre trouvé avec aav5 donc il
  faut une correspondance entre org2 et org3 et des filtres sur les minimums de count") — même
  méthode de régression de Potts (F+J, 8541 features, `fit_weights_potts_from_data`, poids
  inverse-variance `eps=0.5`), filtre = `compte_organoide_adn ≥ T` ET `compte_virus ≥ T` appliqué
  à org2 ET org3 (correspondance entre réplicats + minimum de comptage combinés en un seul
  critère), balayé sur `T ∈ {0,5,10,20,30,50,100}`, fit mutualisé org2+org3 à `T_CHOSEN=20` (repris
  tel quel du choix AAV5, pas encore reconfirmé contre les résultats propres à AAV2). Différence
  clé avec le port : `AAV2_organoides_sorted.csv` (viab) ne garde pas les colonnes organoïde, donc
  ce notebook charge le CSV brut `AAV2_organoides.csv` et redérive en ligne le même filtre
  qualité-viab que `AAV2_viab_sorting.ipynb` (`PLASMID_MIN=1`, `RATIO_MAX=100`) comme population
  de base ; pas de retrait 7m8 (aucun spike-in plasmide trouvé pour AAV2). Exporte
  `lib/aav2_{F,J}_sel_pool_potts_sorted_readoutT20_unreg.npy`. Détail complet dans l'entrée
  `AAV2/selectivity/sorting/AAV2_SEL_potts_readout_depth.ipynb` de "Structure du projet"
  ci-dessus. Logique validée par un smoke-test sur ~12% du CSV brut (pipeline complet exécuté sur
  données réelles échantillonnées) — **notebook lui-même jamais exécuté par Claude dans son
  ensemble** (`feedback_user_runs_notebooks`).
- **2026-09-16 : `AAV2_potts_regression.ipynb` — histogramme de comptages, cache `.npy` pour
  F_potts/J_potts, section 5 mise en veille, nouvelle section 12 (fit CV `plasmid_min∈{20,50}` ×
  `vector>0`) + scatter, section 12b (`ProtocolV3` avec ces poids + réplicats), section 12c :
  décision de GT canonique.**
  1. **Section 1a** (nouvelle, juste après le chargement du CSV) : histogramme du nombre de
     variants par valeur de comptage (`plasmid`/`vector`), bins log-espacés sur les valeurs >0
     (jamais de KDE), fraction de `count==0` en légende — même convention que
     `AAV5_SEL_analysis.ipynb` §1b.
  2. **Section 3 mise en cache** : si `aav2_F_viab_potts.npy`/`aav2_J_viab_potts.npy` existent déjà
     dans `lib/`, la cellule les charge directement au lieu de relancer le fit CV 5-fold complet à
     chaque redémarrage du kernel (`FORCE_REFIT=False` par défaut) — `rank_full`/`info_full`
     valent alors `None`, les cellules avales (courbe CV section 3, ligne `lambda` imprimée
     section 9) gardées en conséquence.
  3. **Section 5 mise entre guillemets** (`""" ... """`, sur demande explicite) : le refit
     held-out 80/20 (un 2e fit CV complet, coûteux) est désactivé par défaut pour que `Run All`
     ne le redéclenche pas — `r_potts`/`pred_potts_test` retombent à `None`, sections 6/9 gardées
     en conséquence. Section 11 (sweep `plasmid_min`×`vector`, `lam=0`) volontairement PAS
     touchée (l'utilisateur y itère encore).
  4. **Nouvelle section 12** : 2 fits Potts avec sélection de λ par CV complète (pas `lam=0` comme
     la section 11) sur `plasmid_min∈{20,50}` × `vector>0` — table `cv_filter_df`
     (n_fit/rank/cv_lambda/r_sorted/r_brut), comparaison contre les fits `lam=0` de la section 11
     si déjà exécutée, courbes CV, et scatter hexbin score-Potts-vs-`target`-réel (population
     filtrée ET population brute).
  5. **Section 12b** : `ProtocolV3` (viab seule) simulé avec CHACUN de ces 2 jeux de poids sur sa
     propre population de fit (`plasmid>=pm & vector>0`) — histogrammes + scatter réel-vs-simulé
     (réutilise `plot_histogram_grid`/`plot_scatter_grid` de la section 8c) ; 5 réplicats (seeds
     différents) par seuil pour comparer la reproductibilité de `r(sim,réel)` au plafond
     déterministe `r_sorted` de la section 12.
  6. **Section 12c — décision utilisateur** : `plasmid_min=50, vector>0` (fit CV de la section 12)
     devient LA GT canonique de viabilité AAV2 pour tout usage `ProtocolV3` en aval,
     **remplaçant** le fit non filtré de la section 3/9 (`aav2_F_viab_potts.npy`, décision du
     2026-09-15) comme défaut. Exporte `aav2_F_viab_potts_plasmid50_vectorpos_cv.npy`/
     `aav2_J_viab_potts_plasmid50_vectorpos_cv.npy` (gitignorés, même convention que les autres
     `.npy` du projet). Le fit non filtré reste en place pour comparaison, pas supprimé ; la
     lignée organoïde IDV (`_sorted_cv`, `_top10k_cv`, HIGH DIVERSITY) n'est pas affectée par
     cette décision. Section 9 amendée d'une note de supersession pointant vers la section 12c.
  Toutes ces cellules validées par smoke-test sur données synthétiques (fit Potts réduit,
  `ProtocolV3` réel sur un petit pool) — **notebook lui-même toujours jamais exécuté par Claude
  dans son ensemble** (`feedback_user_runs_notebooks`).
- **2026-09-15 (suite) : `AAV2_potts_regression.ipynb` — matrice de corrélation 5x5 (+ heatmap)
  ajoutée pour les réplicats de la section 10, et nouvelle section 11 (finale) : sweep de
  régression de Potts sur `plasmid_min` x détection `vector`.** (1) Complète la table de paires
  déjà présente (section 10) par une vraie matrice 5x5 symétrique (`corr_df` + heatmap annotée),
  sur demande explicite ("je voulais savoir la correlation les uns par rapport aux autres aussi").
  (2) Section 11 : refit Potts (`lam=0`, pas de CV — même raison que les sweeps AAV5) sur 10
  combinaisons (`plasmid_min` de `PLASMID_MIN_SWEEP` x `vector>0`/`vector>=0`), poids
  inverse-variance `eps=0.5` recalculés sur chaque sous-ensemble filtré. Motivation du filtre
  `vector>0` : distinguer "vraiment non-viable" de "erreur PCR/synthèse ayant fait dériver l'ADN
  réel du variant désigné" (même phénomène que `new_variant_appearance_analysis.ipynb` sur AAV9),
  `vector==0` ne permettant pas de trancher entre les deux. Chaque fit scoré 2 fois : `r` contre
  sa PROPRE population filtrée ("triée") et `r` contre la population brute complète (53 382
  séquences) — un filtre qui surapprend sur son sous-ensemble sans généraliser au brut se voit
  dans l'écart entre les deux (déjà confirmé par le smoke-test : `r_sorted` sature à 1.000 dès que
  `n_fit < 8541` features, rang déficient — interpolation parfaite, pas un signal réel — alors que
  `r_brut` reste honnête). Logique smoke-testée (matrice sur données factices, sweep sur un
  sous-échantillon de 8 000 lignes) — **notebook toujours jamais exécuté par Claude dans son
  ensemble** (`feedback_user_runs_notebooks`).
- **2026-09-15 (suite) : décision actée — `aav2_F_viab_potts.npy`/`aav2_J_viab_potts.npy`
  (`AAV2_potts_regression.ipynb`, LOW DIVERSITY, `aav_viability_test/aav2.csv` 53 382 séquences)
  deviennent LE modèle de viabilité par défaut pour tester `ProtocolV3` avec AAV2. Les 2 autres
  variantes `aav2_*_potts_{sorted_cv,top10k_cv}.npy` (lignage IDV organoïde `AAV2_organoides*.csv`,
  millions de séquences) sont explicitement étiquetées HIGH DIVERSITY — à n'utiliser QUE pour
  comparer explicitement high-diversity vs low-diversity côté viabilité, jamais par défaut.**
  Écrit noir sur blanc dans la section 9 (export) du notebook, sur demande explicite de
  l'utilisateur. Nouvelle section 10 (finale) ajoutée dans la foulée : (1) plafond GT — r de
  Pearson entre le score Potts déterministe (`score_FJ(seq_matrix, F_potts, J_potts)`, sans aucun
  bruit protocole/NGS) et le `target` réel, jamais calculé ailleurs dans ce notebook (distinct de
  `r_potts` section 5 — fit `idx_train` seul — et de `r_sim` section 8 — une seule simulation
  bruitée) ; (2) 5 réplicats `ProtocolV3` (même `F_potts`/`J_potts`, même librairie complète non
  filtrée 53 382 séquences, même config `D`/`mu`/`T_viab`/`noise_viab` que la section 8 — seed
  différent à chaque fois), table des 10 corrélations de Pearson deux-à-deux entre réplicats +
  histogramme superposé des 5 réplicats vs réel (recalé médiane), même style que la partie 1 d'
  `AAV9_potts_GT_fitting_protocol.ipynb`. Logique smoke-testée sur un sous-échantillon de 4 000
  lignes (r_gt_ceiling + 5 réplicats + 10 paires + histogramme) — **notebook toujours jamais
  exécuté par Claude dans son ensemble** (`feedback_user_runs_notebooks`).
- **2026-09-15 (suite) : `AAV2_potts_regression.ipynb` — grilles d'histogrammes 8c/8d passées en
  échelle Y libre par panneau (au lieu de `sharey=True`) + grilles de scatter réel-vs-simulé
  ajoutées, sur feedback utilisateur ("les courbes les mets pas a la meme echelle on voit plus
  rien apres" puis "remets les scatter plot aussi").** `plot_histogram_grid` (`lib` notebook-local,
  cf. entrée précédente) avait `sharey=True` : les populations filtrées vont de 53 382 à quelques
  milliers de variants avec des étalements très différents, donc un axe Y partagé écrasait tous
  les panneaux sauf celui au pic le plus haut — retiré, chaque panneau a maintenant sa propre
  échelle. Nouveau helper `plot_scatter_grid` (même fichier, même convention `results_by_key`
  keyed par seuil) : un hexbin réel-vs-simulé par seuil avec diagonale y=x, même style que les
  scatters déjà présents en sections 8/8b — appelé juste après `plot_histogram_grid` pour 8c
  (`sweep_results`) et 8d (`sweep_results_sym`), donc chaque sweep a maintenant sa paire
  histogramme+scatter comme 8b. Logique smoke-testée (6 000 lignes, 3 seuils) — **notebook
  toujours jamais exécuté par Claude dans son ensemble** (`feedback_user_runs_notebooks`).
- **2026-09-15 (suite) : `AAV2_potts_regression.ipynb` — histogrammes du sweep 8c manquants
  ajoutés + nouvelle section 8d (filtre symétrique `plasmid` ET `vector`), sur feedback
  utilisateur ("tu n'as pas plot les histogrammes et test un autre filtre... vector").** Refactor
  minimal pour éviter la duplication entre 8c et le nouveau 8d : extraction de deux helpers
  (`run_filtered_protocol(mask)` — sous-échantillonne, calibre `D` sur les comptages réels
  filtrés, lance un round `ProtocolV3`, retourne `d0`/`r_sim_real`/les tableaux bruts nécessaires
  au tracé ; `plot_histogram_grid(results_by_key, ...)` — une grille de panneaux réel-vs-simulé,
  un par seuil) insérés juste après le markdown de la section 8c, PAS de refactor de la section 8
  elle-même (reste celle réglée à la main par l'utilisateur, `D=2e8` au moment de cette note).
  8c (sweep `plasmid_min`, déjà exécuté avec succès par l'utilisateur — r(sim,réel) montait
  +0.505→+0.547 de `plasmid_min=0` à `20`) réécrit pour utiliser le helper et stocker les tableaux
  complets par seuil (pas seulement le résumé scalaire comme avant), plus une nouvelle cellule
  grille d'histogrammes juste après le tableau/courbe existants. Nouvelle section 8d : sweep
  IDENTIQUE mais avec le masque symétrique `(plasmid>=min) & (vector>=min)` au lieu de
  `plasmid>=min` seul — table + courbe superposée sur le MÊME graphe que 8c (comparaison directe
  seuil-à-seuil des deux stratégies de filtre) + sa propre grille d'histogrammes. Smoke-test sur
  6 000 lignes : le filtre symétrique donne un `r` nettement meilleur que le filtre plasmide seul
  au même seuil (~0.85-0.87 vs ~0.71-0.75 à `min=3-5` sur cet échantillon réduit — pas
  nécessairement représentatif de la pleine échelle, à confirmer par l'utilisateur sur les
  53 382 lignes). **Notebook toujours jamais exécuté par Claude dans son ensemble**
  (`feedback_user_runs_notebooks`) — seul le smoke-test isolé (6 000 lignes, hors notebook) a
  tourné côté Claude.
- **2026-09-15 (suite) : `AAV2_potts_regression.ipynb` — sections 8b/8c ajoutées, filtre
  `plasmid_min` sur le test `ProtocolV3` + sweep de seuil, toujours pendant que l'utilisateur
  exécutait le notebook.** Sur demande explicite : (1) 8b répète le test `ProtocolV3` de la
  section 8 mais retire d'abord les variants `plasmid < PLASMID_MIN` (défaut 3) — DE LA LIBRAIRIE
  SIMULÉE ET DE LA POPULATION RÉELLE DE COMPARAISON, pour rester cohérent (histogramme/scatter de
  la section 8 comparaient déjà le simulé au `target` réel, mais sur les 53 382 lignes complètes,
  pas la population filtrée que la simulation représente désormais) ; `D` recalibré sur le ratio
  reads/variant RÉEL de CETTE population filtrée (`D=(sum(plasmid)+sum(vector))/2` sur les lignes
  filtrées, remplace le nombre magique codé en dur de la section 8 par un calcul dérivé des
  données — reads/variant réel ~23→61 (plasmide) et ~30→79 (vecteur) en passant du dataset complet
  à `plasmid>=3`). (2) 8c généralise en sweep sur `plasmid_min ∈ {0,3,5,10,20}` (0 = cas non
  filtré, avec son propre `D` recalculé — PAS la valeur actuellement réglée à la main par
  l'utilisateur en section 8, qui reste intouchée), une simulation `ProtocolV3` complète par
  seuil, table + courbe de `r(simulé, réel)` en fonction du seuil. Section 8 elle-même (baseline
  non filtré, `D=1.5e8` actuellement réglé par l'utilisateur) **non modifiée** — 8b/8c ajoutées à
  la suite, pas en remplacement. Insertion chirurgicale via `NotebookEdit` (le notebook restait
  ouvert/en cours d'exécution — CV de la section 5 toujours à 80% au moment de l'ajout, sections
  8/8b/8c pas encore exécutées). Logique validée par un smoke-test sur un sous-échantillon de
  6 000 lignes (fit Potts réduit + un run `ProtocolV3` filtré complet + un mini-sweep à 3 seuils)
  — **notebook toujours jamais exécuté par Claude** (`feedback_user_runs_notebooks`).
- **2026-09-15 (suite) : `AAV2_potts_regression.ipynb` — section "1b. Reads per step
  (checkpoint)" ajoutée juste après le chargement du CSV (avant toute pondération/régression),
  pendant que l'utilisateur exécutait déjà le notebook.** Table `total_reads`/`reads par variant`
  (moyenne+médiane)/`n_variants>0`/`fraction>0` pour les 2 checkpoints du dataset (`plasmid` =
  librairie initiale, `vector` = post-sélection viabilité — mêmes rôles que `lambda0p`/`lambda2p`
  dans le framework `Protocol` de la section 8), plus `number_of_seq_threshold` (`lib/analysisV1.py`,
  réutilisé tel quel) pour les seuils [1,10,100,1000]. Insertion chirurgicale via `NotebookEdit`
  (PAS de réécriture complète du fichier comme pour les révisions précédentes) — le notebook étant
  en cours d'exécution (CV de la section 5 à 80% au moment de la demande), reconstruire tout le
  JSON aurait risqué d'écraser les sorties déjà calculées en cas de sauvegarde concurrente côté
  utilisateur. Seul autre changement : `number_of_seq_threshold` ajouté à l'import `analysisV1` de
  la cellule Setup (section 0) — pas d'autre cellule touchée. **Note en marge, non corrigée** :
  l'utilisateur a modifié `D` dans la section 8 (`ProtocolV3` test) de `1.5e6` à `1.5e8` en cours
  de route, mais le commentaire du code dit encore "~230-300 reads/variant" (qui correspondrait à
  `D≈1.5e7`, ni à `1.5e6` ni à `1.5e8`) — valeur finale toujours en cours d'expérimentation par
  l'utilisateur au moment de cette note, pas swept ni figée ici.
- **2026-09-15 (suite) : `AAV2_potts_regression.ipynb` révisé sur feedback utilisateur — naïf
  abandonné, split 80/20, analyse des poids F/J (heatmaps), test `ProtocolV3`.** Quatre
  changements demandés explicitement : (1) baseline naïf group-means retiré entièrement (section
  4 originale) — "on sait que le naive ne va pas on s'en fout" ; (2) split held-out 50/50 → 80/20
  (`train_size=0.8`), plus de contrainte de symétrie avec un baseline à comparer ; (3) nouvelle
  section 4 "F/J weight analysis" — `plot_teacher_weights` (`lib/analysisV1.py`, réutilisé tel
  quel) pour F + mean(J), plus un panneau mean(|J|) (le mean(J) brut peut masquer la magnitude si
  les signes s'annulent dans une cellule) et les 4 paires de positions les plus fortement couplées
  en heatmap 20×20 individuelle — même diagnostic qu'`AAV9_potts_GT_score_study.ipynb` ; (4)
  nouvelle section 8 "`ProtocolV3` test" — run viabilité seule (`F_sel=J_sel=0`) sur la librairie
  réelle complète (53 382 séquences, pas de sous-échantillonnage), `lambda0` calibré sur le vrai
  `plasmid` de chaque séquence, `D=1.5e6` calibré pour matcher le ratio reads/variant réel
  (~23-30, `D/d0=28.1`) — même recette que `AAV5_viab_fitting_protocol.ipynb` (`rho=1e-3, mu=50,
  T_viab=1.0, noise_viab=0.5, dilution_factor=1e5`, non recalibrés spécifiquement pour ce
  dataset) ; histogramme recalé médiane (réel vs simulé superposés) + hexbin réel-vs-simulé,
  Pearson r rapporté (invariant au recalage). **Grille lambda custom conservée sur demande
  explicite** : `lambdas_grid=np.logspace(-2, 5, 20)` (0.01 à 100 000, au lieu du défaut de la
  fonction `np.logspace(-1, 2, 30)`) — appliquée aux deux fits du notebook (section 3 full-data ET
  section 5 held-out 80/20) pour rester comparables entre elles. Logique des nouvelles sections
  validée par un smoke-test sur un sous-échantillon de 4 000 lignes (heatmaps, split 80/20, un
  round `ProtocolV3` complet) — **notebook lui-même toujours jamais exécuté**
  (`feedback_user_runs_notebooks`).
- **2026-09-15 (suite) : nouveau notebook `AAV2_potts_regression.ipynb`, GT Potts pour AAV2 basée
  sur `Modelization_V1/notebooks/aav_viability_test/aav2.csv` (53 382 séquences, PAS le CSV
  organoïde IDV `AAV2_organoides.csv` déjà utilisé ailleurs dans `AAVs dataset/AAV2/`), demandé
  explicitement par l'utilisateur ("même type de retrieval que pour fit4function").** Copié
  self-contained dans `AAVs dataset/AAV2/viability/aav2.csv` (gitignoré comme `aav9.csv`/
  `fit4functionaav9.csv`, même convention que la migration V1→V2 du 2026-08-31) plutôt que lu
  cross-repo depuis `Modelization_V1`. Méthode : `RegressionV1.fit_weights_potts_from_data`
  (même fonction que `AAV9_potts_regression.ipynb` et la Part A d'`AAV9_fit4function_potts_vs_mlp.
  ipynb`) — full-data CV, comparaison held-out à un baseline group-means local (50/50,
  `random_state=0`), percentile recovery, check de crédibilité brute-force top-500, export.
  **Écart déterminé par rapport au docstring de la fonction** : celui-ci suggère
  `sample_weight=1/error**2` pour aav2.csv/aav5.csv (colonne `error` réelle, contrairement à la
  constante 0.1 d'aav9.csv) — vérifié puis écarté, `1/error**2` est dominé par une poignée de
  lignes (top 100/53 382 lignes = 45% du poids total, top 1000 = 84%), même pathologie que
  `AAV5_SEL_profile_model_sel_org2_invvar.ipynb` (déjà abandonné dans ce projet pour la même
  raison). Utilisé à la place : la formule inverse-variance standard du projet sur les comptages
  bruts `plasmid`/`vector` (`eps=0.5`, top 100 lignes = 11% du poids, top 1000 = 43% — bien mieux
  réparti), justifié aussi par le fait que `target` reconstruit quasi exactement comme
  `log2((vector+0.5)/(plasmid+0.5))` (r=+0.987) — `target` EST déjà ce log enrichment `eps=0.5`
  standard du projet, donc pondérer par ces mêmes comptages est le choix cohérent. Path resolution
  vers `lib/` via le pattern robuste (remontée jusqu'à `Modelization_V2/`) déjà utilisé partout
  dans `Selectivity/AAV{2,5}` — PAS le `../../lib` en dur des notebooks AAV9 plus anciens
  (`AAV9_potts_regression.ipynb`/`AAV9_potts_GT_score_study.ipynb`), un bug déjà documenté
  ailleurs dans ce fichier (ne pointe plus nulle part depuis leur emplacement actuel, import
  silencieux du lib pip-installé de V1) — pas corrigé dans ces notebooks-là, juste évité dans le
  nouveau. Export `lib/aav2_F_viab_potts.npy`/`aav2_J_viab_potts.npy` (SANS suffixe, miroir exact
  de `aav9_F_viab_potts.npy` — le GT canonique construit directement depuis l'aav2.csv éponyme,
  distinct de `aav2_F_viab_potts_sorted_cv.npy` qui vient du CSV organoïde IDV). Logique validée
  par un smoke-test sur un sous-échantillon de 4000 lignes (grille lambda réduite, pas de plot) —
  **notebook lui-même jamais exécuté**, conforme à `feedback_user_runs_notebooks` (l'utilisateur
  lance lui-même les notebooks coûteux).
- **2026-09-15 : réorganisation des notebooks actifs `Selectivity/AAV2` et `Selectivity/AAV5` en
  sous-dossiers `viability/`/`selectivity/` × `sorting/`/`analysis of noise/`/`analysis of
  recovery/`, sur demande explicite de l'utilisateur.** `git mv` (historique préservé), CSV laissés
  en place à la racine d'AAV2/AAV5 (données partagées, déjà résolues par un fallback
  location-independent dans chaque notebook — vérifié, zéro code modifié), `AAV5/obsolete/` non
  touché. Mapping complet ancien chemin → nouveau chemin dans l'entrée `notebooks/notebooks/
  Selectivity/` de "Structure du projet" ci-dessus.
- **2026-09-14 (suite) : plus de dithering à l'avenir (feedback utilisateur) ; 3 nouveaux notebooks
  AAV2 écrits (fitting_protocol, profile_model, top10k Potts+Protocol+MLP) mais AUCUN exécuté —
  l'utilisateur lance lui-même les notebooks coûteux depuis cette session.** Deux préférences
  actées en mémoire : (1) `feedback_no_dithering` — le dithering (jitter sur les comptages avant de
  recalculer un ratio) "ajoute juste du bruit", ne plus l'utiliser (les notebooks déjà écrits avec
  le gardent, pas de retrait rétroactif) ; (2) `feedback_user_runs_notebooks` — préparer les
  notebooks lourds (entraînement/CV/simulation) sans les exécuter automatiquement via nbconvert,
  laisser l'utilisateur déclencher l'exécution. Nouvelle population de travail introduite dans
  `AAV2_viab_top10k_potts_protocol_mlp.ipynb` : les 10 000 variants au plus fort `compte_plasmide`
  (représentation la plus fiable de la librairie initiale, plasmide dans [19,61]) plutôt qu'un
  sous-échantillon aléatoire — choisie justement pour rendre la discrétisation négligeable sans
  dithering. Détail complet dans les 3 entrées dédiées d'`AAV2/` dans "Structure du projet"
  ci-dessus.
- **2026-09-14 (suite) : premier notebook AAV2 (`AAV2_viab_sorting.ipynb`) — pas de bimodalité
  fit/non-fit comme AAV5/AAV9, mais un meilleur r held-out de régression Potts (+0.266).** Même
  méthode que la session AAV5 (dithering, sweep informé par les données, régression de Potts),
  appliquée à `AAV2_organoides.csv` (4 273 463 lignes) sur demande explicite ("refais pareil").
  Dataset structurellement différent : comptages ENTIERS (pas des multiples de 0.5), `compte_plasmide`
  plafonne à 61 (médiane=1) contre 500+ pour AAV5, aucun spike-in identifiable comme le 7m8 mais une
  contamination virale diffuse (jusqu'à 1 081 060 reads virus pour ≤9 reads plasmide, log2
  enrichissement jusqu'à +19.24 — un artefact technique probable, type index-hopping, en continuum
  lisse donc géré par un cap de ratio plutôt qu'une coupure hamming). Dithering + sweep de
  `PLASMID_MIN` (0 à 20) : **aucune bimodalité ne ressort à aucun seuil testable** — constat honnête
  (le plafond de comptage 10x plus bas qu'AAV5 ne permet pas d'atteindre le régime informatif).
  Filtre retenu : `PLASMID_MIN=1` + `RATIO_MAX=100` → 2 746 425 lignes (64.3%),
  `AAV2_organoides_sorted.csv`. Régression de Potts (même recette qu'`AAV5_SEL_analysis.ipynb`,
  poids inverse-variance eps=0.5) : **held-out r=+0.266 (CV, lambda=1e3 intérieur à la grille) —
  meilleur que la viabilité AAV5 (~0.2)**, malgré un dataset a priori plus bruité. Exporté
  `lib/aav2_{F,J}_viab_potts_sorted_cv.npy`. Deux plantages mémoire rencontrés et corrigés en route
  (refit sur un design Potts dense trop grand, 210k×8541 lignes) — poids finaux = ceux du fit sur
  90 000 lignes déjà validé held-out, pas de refit sur plus de données (même convention qu'AAV5).
  Détail complet dans l'entrée `AAV2/AAV2_viab_sorting.ipynb` de "Structure du projet" ci-dessus.
- **2026-09-14 (suite) : à `D` calibré sur le ratio reads/variant réel (D=1e6, D/d0=20), le protocole
  simulé sur la phase viabilité perd la bimodalité réelle — résultat inattendu, piste ouverte.**
  Nouveau notebook `AAV5_viab_fitting_protocol.ipynb` : `ProtocolV3` (viab seule, `F_sel=J_sel=0`) sur
  50 000 variants d'`AAV5_organoides_sorted.csv` (le CSV validé juste avant), `lambda0` calibré sur le
  VRAI `compte_plasmide` de chaque variant. `D=1e6` choisi pour matcher le ratio reads/variant global
  réel (~21-39 selon le CSV) à cette échelle réduite — au lieu du `D=1e8` (100x trop dense) repris
  d'AAV9 dans les notebooks précédents. Sanity check `r(score GT, réel)=+0.201` cohérent avec les
  chiffres déjà connus. Mais `r(sim_viab, réel)` chute à `+0.099` (+0.092 après dithering) — bien en
  dessous des +0.196/+0.266 obtenus à `D=1e8`. Dithering (même technique qu'`AAV5_viab_sorting.ipynb`,
  quantum réel=0.5/simulé=1) confirme que ce n'est pas un artefact de visualisation : la distribution
  réelle ditherée reste bimodale (2 pics, -2.28/+0.70) mais la SIMULÉE devient unimodale (1 pic,
  +0.18) — le bruit de mesure simulé (NGS `Multinomial` + `noise_viab=0.5`) à cette profondeur noie
  le signal déterministe F+J, alors que le signal biologique réel reste visible dans les vraies
  données à ce même ordre de grandeur de comptage. Accord réplicat-réplicat simulé seulement +0.35.
  **Mise à jour (même jour) : refit `F_viab`/`J_viab` directement sur ces 50 000 variants** (au lieu
  des poids globaux fit sur 3.8M lignes), pour trancher entre "décalage poids/échantillon" et "bruit
  de protocole". Résultat : le fit local généralise MOINS bien (r held-out sur split interne 25k/25k
  = +0.152, pire que le +0.201 du fit global sur ce même échantillon — design rank-déficient à cette
  échelle, 4842/8541). Avec les poids refit sur les 50 000 en entier (le GT le plus favorable
  possible pour cet échantillon), `r(sim_viab, réel)` remonte à +0.179 (+0.171 dithéré) — mieux que
  +0.099, mais **la distribution simulée ditherée reste UNIMODALE** (1 pic) alors que la réelle
  reste bimodale (2 pics, inchangé). **Conclusion renforcée : c'est bien le bruit de mesure simulé à
  `D=1e6`, pas un décalage de poids GT, qui détruit la bimodalité.** Piste ouverte, toujours pas
  faite : recalibrer `noise_viab`/`T_viab`/`rho`/`mu` spécifiquement pour AAV5 à ce régime de
  profondeur. Détail complet dans l'entrée `AAV5/AAV5_viab_fitting_protocol.ipynb` de "Structure du
  projet" ci-dessus.
- **2026-09-14 (suite) : le pic "parasite" sur `sim_viab` dans `AAV5_SEL_fitting_protocol_org2org3.ipynb`
  n'est pas un artefact — c'est une vraie population non-fit, correctement simulée ; le vrai problème
  est que le `y_viab` réel comparé est déjà conditionné sur succès organoïde.** Correction utilisateur
  sur mon diagnostic précédent (que j'avais qualifié à tort d'"artefact de pileup au plancher du
  pseudocount"). Vérifié : `AAV5_organoides_sorted.csv` (3 822 400 variants, filtre `viab` seul, PAS
  conditionné organoïde) est bien bimodal en `log2_enrichissement_virus_sur_plasmide` (mode non-fit
  ~-2.9, mode fit ~+0.9/+1.4) — même structure que `fit4functionaav9.csv`. Le CSV `_sel_org2org3.csv`
  utilisé pour la comparaison de ce notebook conditionne sur `compte_organoide_{2,3}_adn > 0`, ce qui
  exclut structurellement le mode non-fit côté réel (un signal organoïde suppose que le virus a
  d'abord été produit : `compte_virus` y est strictement >0 sur les 50 000 lignes vérifiées) — donc
  comparer ce `y_viab` (survivant only) au `sim_viab` complet (bimodal) n'est pas une comparaison
  juste. Nouveau notebook `AAV5_viab_sorting.ipynb` (exécuté) : reprend le filtre `viab` sur la
  population complète en 2 étapes contrôlées (7m8 seul, puis retrait des seuls comptages nuls) plutôt
  que le bornage `[10,500]` à tâtons de l'ancien filtre — `compte_virus` n'est JAMAIS nul dans tout le
  CSV brut (5 595 543 lignes), donc "retirer les 0 counts" ne filtre que `compte_plasmide == 0`
  (9.84%, exactement la fraction déjà non-finie de la colonne fournie — n'ajoute rien à `isfinite()`).
  Résultat : 5 044 820 variants conservés (90.2%) contre 3 822 400 (68.3%) pour l'ancien filtre, mais
  distribution plus fragmentée (6 pics vs 4, moins bien séparés en 2 clusters).
  **Piste tranchée par dithering** (ajouté au même notebook, même tour) : `compte_plasmide`/
  `compte_virus` sont TOUJOURS des multiples exacts de 0.5 (100% des lignes), et en dessous de
  `compte_plasmide < 5` il n'existe que 10 valeurs de comptage possibles dans tout le CSV — d'où les
  pics fins (des centaines de milliers de variants sans rapport partagent le même couple de
  comptages, donc le même ratio, par manque de résolution). Ajout d'un bruit `Uniform(-0.25,+0.25)`
  aux comptages avant de recalculer le ratio (`eps=0.5`, visualisation uniquement, ne remplace rien
  en aval) : sur la population "comptages nuls seuls retirés" (`df_nz`, 5 044 820 lignes), le
  dithering fait ENTIÈREMENT disparaître la bimodalité — une seule bosse large, unimodale. Sur
  l'ancien filtre `[10,500]` (3 822 400 lignes), la bimodalité SURVIT au dithering (2 pics nets,
  vallée franche) — donc pas un artefact là. Sweep de `PLASMID_MIN` (0/2/5/8/10/15/20/30, dithering
  à chaque seuil, "vallée/pics" = profondeur de la vallée entre les 2 modes, 1=unimodal, →0=séparé) :
  unimodal jusqu'à 5, bimodalité apparaît à 8 (vallée=0.699) et se renforce à 10 (0.630) — puis
  rendements décroissants (15→0.533, 20→0.495, 30→0.447 pour une perte de diversité sévère,
  3.82M→1.21M). **Conclusion : l'ancien `PLASMID_MIN=10` n'était pas arbitraire — il tombe quasiment
  exactement sur le seuil où la bimodalité réelle devient robuste ; en dessous, la population est
  dominée par du bruit de comptage qui noie tout signal fit/non-fit, même une fois la discrétisation
  corrigée.** Recommandation retenue : garder `AAV5_organoides_sorted.csv` comme CSV `viab` de
  référence, ne PAS le remplacer par `AAV5_organoides_sorted_viab.csv` (écrit à titre exploratoire
  seulement, insuffisant). Détail complet dans l'entrée `AAV5/AAV5_viab_sorting.ipynb` de "Structure
  du projet" ci-dessus.
- **2026-09-14 : meilleur résultat AAV5 `sel_org2` de la session (r held-out +0.596), via un
  filtre par intersection de réplicats plutôt qu'un seuil de comptage.** Nouveau notebook
  `Modelization_V2/notebooks/notebooks/Selectivity/AAV5/AAV5_sel_sorting.ipynb` : filtre **`sel`**
  (pendant du filtre existant, renommé **`viab`**) qui borne `compte_virus` (dénominateur de
  `sel_org_i`, même rôle que `compte_plasmide` pour `viab`) ; extension avec un seuil sur
  `compte_organoide_2_adn` ; puis, motivé par le constat qu'un seuil de magnitude ne peut pas
  distinguer "signal faible réel" de "sous-échantillonné", un **filtre par intersection de
  réplicats** (`compte_organoide_2_adn > 0` ET `compte_organoide_3_adn > 0`, moins le 7m8) — même
  logique que `AAV5_SEL_potts_readout_depth.ipynb` mais sans seuil de magnitude à choisir.
  `ShallowProfileMLP` entraîné sur les 4 variantes dans
  `AAV5_SEL_profile_model_sel_sorting.ipynb` : brut +0.429, sel(virus) +0.430, sel_org2(virus+org2)
  +0.453, **sel_org2∩org3 +0.596** (n_fit=62 300, 30% du volume du filtre dur précédent). Une
  piste alternative (pondération inverse-variance dans la loss MSE, au lieu d'un filtre dur) a été
  testée et **écartée** — dégradait `r` partout car dominée par quelques poids extrêmes issus de
  comptages de contamination ; détail dans la section AAV5 de "Structure du projet" ci-dessus
  (`AAV5_SEL_profile_model_sel_org2_invvar.ipynb`, notebook depuis supprimé). Au passage :
  `AAV5_SEL_deep_profile_model_sel_org2.ipynb` (DeepProfileMLP vs ShallowProfileMLP) supprimé du
  dossier actif par l'utilisateur — Deep systématiquement un peu pire que Shallow sur `sel_org2`
  (Δr ≈ -0.011/-0.012), d'où l'abandon de la variante deep pour la suite. Deux CSV dérivés
  obsolètes supprimés (`AAV5_organoides_classements.csv`, `AAV5_organoides_sorted_organoide.csv` —
  plus référencés par aucun notebook actif).
- **2026-09-09 : `fit_weights_potts_from_data` accepte un `lam` fixe (RegressionV1.py V2).**
  `lam=None` (défaut) garde la sélection par CV ; `lam=0` fait un fit minimum-norm non
  régularisé (lstsq SVD, `fit_weights_potts_unregularized` — qui accepte maintenant aussi
  `sample_weight`, lstsq pondéré par `sqrt(w)`) ; `lam>0` fitte à cette pénalité sans CV. Dans
  ces deux derniers cas `info["cv_mse"]`/`info["lambdas_grid"]` valent `None`. Motivation :
  dans `AAV5_SEL_analysis.ipynb` (section 3), la CV tombait systématiquement sur des λ énormes
  (1e3–2e5) qui rétrécissent F/J vers la moyenne et tuent le signal → le notebook passe
  maintenant `LAM=0.0`. Message du module `RegressionV1.py` (V2 uniquement) bumpé 1.4→1.5. La
  copie `Modelization_V1/lib/RegressionV1.py` n'est PAS modifiée. **Système d'export des poids
  AAV5 changé** : `AAV5_SEL_analysis.ipynb` écrit désormais `aav5_{F,J}_{name}_potts_{tag}.npy`
  avec `tag` ∈ {`unreg`, `lam<x>`, `cv`} dérivé de `LAM` — les fichiers historiques
  `aav5_{F,J}_{name}_potts.npy` (λ CV) ne sont plus écrasés. `.gitignore` élargi
  `aav5_*_potts.npy`→`aav5_*_potts*.npy` (idem aav2) pour couvrir le suffixe de tag.
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
