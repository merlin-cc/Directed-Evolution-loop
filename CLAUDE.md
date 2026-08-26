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
│   │   ├── RegressionV1.py              # construction de datasets pour ridge regression
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
│   │                                    #   ProtocolCrossPackagingMechanistic : pool de cellules physiques partagé,
│   │                                    #   mais PAS tractable à l'échelle N1 du projet (jusqu'à ~1e10 dans certains
│   │                                    #   sweeps) — plafonnée à MAX_MECHANISTIC_CELLS=5M, lève une erreur au-delà ;
│   │                                    #   gardée comme esquisse de la structure du problème, PAS fonctionnelle en
│   │                                    #   l'état (chaque cellule n'a qu'un seul occupant — la redistribution
│   │                                    #   multi-occupants, cœur du cross-packaging, reste un TODO).
│   ├── notebooks/
│   │   ├── analysis of correlation/     # F_permutation_recovery_correlation.ipynb : F_viab AAV9 réel, J_viab=J_sel=0
│   │   │                                #   (pas d'épistasie pour ce premier passage) ; F_sel = 10 permutations des
│   │   │                                #   lignes (axe acide aminé) de F_viab (une par clé jax), corrélation GT
│   │   │                                #   avec F_viab mesurée ; pool de séquences + split train/val/test fixes
│   │   │                                #   pour les 10 runs ; un ProfileMLP entraîné par permutation sur le log
│   │   │                                #   enrichment de sélectivité (target2), F_sel_hat recouvré par scan
│   │   │                                #   single-mutant, comparé au F_sel vrai de ce run
│   │   ├── analysis of parameters for viability/  # Protocol_parameters_and_first_classic_use.ipynb ; mu_HEK_multiplicity_sweep.ipynb
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
│   │   ├── directed_evolution_loop/     # DE_loopV1.ipynb — boucle de simulation d'évolution dirigée bout-en-bout
│   │   ├── selectivity_weight_regimes/  # trio F_sel/J_sel corrélé/anticorrélé/indépendant + variantes (bilinear head,
│   │   │                                #   double mutant designed, profile-only) + CSV de diversité mis en cache
│   │   │                                #   MLP_viability_noise_denoising.ipynb : viabilité SEULE (GT F_viab/J_viab
│   │   │                                #   régularisé), sweep de noise_viab à pool de séquences fixe pour tester si
│   │   │                                #   le MLP débruite (corrélation prédiction vs score vrai vs. label NGS brut)
│   │   ├── aav_viability_test/          # AAV{2,5,9}_profile_model.ipynb (entraîne ProfileMLP sur données réelles),
│   │   │                                #   AAV_MLP_weights_recovery.ipynb, checks de recouvrement d'erreur/top500
│   │   │                                # AAV9_fitting_protocol.ipynb (2026-08-25) : calibre les hyperparamètres du
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
│   │   ├── reproductibility/            # log_enrichment_histograms.ipynb (2026-08-26) : lit fit4functionaav9.csv
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
│   │   │                                #   aav_viability_test/AAV9_profile_model.ipynb, entraîné sur 20% des variants
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
│   │   └── deeper_mlp/                  # diversity_sweep_deeper_mlp.ipynb : proche de diversity_sweep.ipynb mais PAS
│   │                                    #   identique — mu=10/rho=1e-4/N0=150*N1 alignés, mais D=5e8 fixe (pas 1e8) et
│   │                                    #   noise_viab=3 (pas 0.5, écart non résolu — cf. "État actuel") ; grille d0 propre
│   │                                    #   200 à 200k (sur-ensemble du 5k-200k de diversity_sweep.ipynb côté petit d0)
│   │                                    #   mais compare CETTE FOIS deux architectures entraînées sur les mêmes données :
│   │                                    #   ShallowProfileMLP (~27k params, l'archi standard du projet, juste renommée)
│   │                                    #   vs DeepProfileMLP (~90k params, nouveau) — embedding par position partagé
│   │                                    #   (Linear 20->16 + GELU) moyenné (avg pool) sur les 7 positions pour la branche
│   │                                    #   "profile", + branche pairwise (MLP sur les 21 paires de positions concaténées,
│   │                                    #   avg pool) plus expressive que le BilinearHead déjà testé (et jugé marginal)
│   │                                    #   dans MLP_bilinear_head_anticorrelated.ipynb, + tête dense à 4 couches (256-
│   │                                    #   128-64-32) au lieu de 2 ; même batch_size adaptatif pour isoler l'architecture
│   │                                    #   comme seule variable ; inclut les mêmes diagnostics d'entraînement
│   │                                    #   (best_epoch/total_steps/val_mse) que diversity_sweep.ipynb pour vérifier si le
│   │                                    #   modèle profond profite réellement de plus de gradient steps, pas seulement de
│   │                                    #   plus de capacité. Recovery top-1000 comparée sur le pool d'éval fixe avec DEUX
│   │                                    #   références GT<->protocole : profondeur fixe (constante ~93%, section 4) et
│   │                                    #   profondeur APPARIÉE (D_eval_matched = D_FIXED/d0 * 50 000, ré-simulée à chaque
│   │                                    #   d0 — le vrai point de comparaison "juste", puisque les labels d'entraînement du
│   │                                    #   MLP sont eux-mêmes à cette profondeur D_FIXED/d0, pas à la profondeur fixe du
│   │                                    #   pool d'éval) ; graphe recovery sur le split interne au sweep supprimé (sa
│   │                                    #   taille dépend de d0, donc pas comparable d'un point à l'autre)
│   ├── docs/                            # PDF/tex de référence (extraction de poids, encodage one-hot, protocole, ...)
│   └── contrib/                         # export Colab autonome d'un collaborateur, non importé ailleurs
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
