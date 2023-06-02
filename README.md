# MD2 @ Dabiskās valodas apstrāde | Māris Kalniņš, mk20126
> LU DF, 3. kurss, m.g. 2022/2023

## Par Projektu
## Izvēles Dati
## Komandas
1) `cd` projektu mapē
2) `python -i nb_experiment.py` uzsāk Python skriptu
3) `start(“data.tsv”)` *var izmantot arī citu file name* pārliek datus (klases + teksts) no .xml formāta uz .tsv tālākai apstrādei. Šajā solī papildus izveido `frequency.tsv` failu, kur uztur tokenu absolūto biežumu no visu ierakstu tekstiem. Tokeni tiek standartizēti, bet netiek lemmatizēti. Biežums tiek kārtots dilstošā secībā.
4) `initialise(“stoplist.txt”, “frequency.tsv”)` ignorē visbiežāk sastopamos nesvarīgos vārdus (kā saikļi, palīgvārdi u.c.), kuri tiek saglabāti `stoplist.txt`
5) `train_classifier(“data.tsv”)` uztrenē Naivo Bayes klasifikatoru (no NLTK bibliotēkas) uz padotajām klasēm un tekstiem. 
6) `run()` lietotājs spēj ievadīt tekstu, kurš tiek klasificēts un izvada visu klašu varbūtībau, ka šis padotais lietotāja ievads pieder tai klasei.
7) `run_validation(“data.tsv”, 5, 1)` izveido confusion matricu un salīdzina klasifikatora precizitāti.![image](https://github.com/bezgaliba/MD2/assets/74833724/e169c2ab-78d6-4b05-94b4-dabc126ee67a)

## Secinājumi
