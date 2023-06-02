# MD2 @ Dabiskās valodas apstrāde | Māris Kalniņš, mk20126
> LU DF, 3. kurss, m.g. 2022/2023

## Par Projektu
  Otro mājasdarbu ietvarā, izmantojot lekcijas sniegto kodu un chatgpt asistenci izstrādē, tika izveidots klases atpazinējs. Klase ir kategorija, kuru vārdu iedala. Par cik tika ņemti medicīnas nozares pacientu izmeklēšanas dati, tad šis projekts klasificēs datus medicīnas kontekstā. Ir arī izveidota klašu matrica, izmantojot testa un treniņa dati. Tika izmantota viena datu kopa gan priekš treniņiem, gan priekš testa. Priekš trenēšanas tika izmantoti random 60% data, priekš treniņa - 40% (kodā var nomainīt rindu `train_ratio=0.6`, lai mainītu šo attiecību).
## Izvēles Dati
  Dati tika atlasīti no publiski pieejamās datu kopas “LVMED” - http://hdl.handle.net/20.500.12574/67, kas arī pievienots šī projekta .xml formāta failā `LVMED-Transcripts-900.xml`, kas attiecīgi arī tiks apstrādāts ar .py skriptu.
## Komandas
1) `cd` projektu mapē
2) `python -i nb_experiment.py` uzsāk Python skriptu
3) `start(“data.tsv”)` (*var izmantot arī citu file name*) pārliek datus (klases + teksts) no .xml formāta uz .tsv tālākai apstrādei. Šajā solī papildus izveido `frequency.tsv` failu, kur uztur tokenu absolūto biežumu no visu ierakstu tekstiem. Tokeni tiek standartizēti, bet netiek lemmatizēti. Biežums tiek kārtots dilstošā secībā.
4) `initialise(“stoplist.txt”, “frequency.tsv”)` ignorē visbiežāk sastopamos nesvarīgos vārdus (kā saikļi, palīgvārdi u.c.), kuri tiek saglabāti `stoplist.txt`
5) `train_classifier(“data.tsv”)` uztrenē Naivo Bayes klasifikatoru (no NLTK bibliotēkas) uz padotajām klasēm un tekstiem. 
6) `run()` lietotājs spēj ievadīt tekstu, kurš tiek klasificēts un izvada visu klašu varbūtībau, ka šis padotais lietotāja ievads pieder tai klasei.
7) `run_validation(“data.tsv”, 5, 1)` izveido confusion matricu un salīdzina klasifikatora precizitāti. `5` var aizstāt ar moduļu trenēšanas reizēm. Jo lielāka šī vērtība, jo lielāka būs precizitāte, jo vairāk aizņems PC resursus.

## Secinājumi | Rezultātu novērtēšana

Uztrenēto modeļu precizitāte:
  ![image](https://github.com/bezgaliba/MD2/assets/74833724/3625e5e0-1f77-42a0-9a5f-5e32c975a5f8)
Nolasot no attēla rādītājus, mana uztrenēta modeļa veiktspējas precizitāte ir ļoti zema, nozīmē, ka mēdz tekstu klasificēt biežāk nekorekti, nekā korekti. Lai uzlabotu šo rādītāju, īpaši priekš latviešu valodas, būtu jāpārskata 'feature engineering' paradokss.
Confusion matricas modelis, izmantojot 5-fold:
![image](https://github.com/bezgaliba/MD2/assets/74833724/4ea7927d-e0a5-4fc4-8ac8-99278915e34a)

