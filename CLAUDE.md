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
│   │   └── aav9_{F,J}_viab_mlp.npy      # gitignorés — artefacts dérivés, régénérés via AAV9_profile_model.ipynb
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
