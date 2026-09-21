# Modellkontroll

Dobbelklikk `KJØR_FULL_MODELLKONTROLL.bat` for å kjøre hele, reproducerbare testen.

Den vurderer `best_model.pt` mot NIH sin offisielle holdt-utenfor-testliste. Det er 25 596 bilder, og resultatene skrives til `audit_results`.

Siste fullførte kontroll ble kjørt 21. september 2026 med GPU. Den ga gjennomsnittlig AUROC på 0,795. Det er et mål på om modellen kan rangere positive bilder over negative bilder, ikke et mål på om den kan brukes til diagnose.

Kalibreringen er for svak for en offentlig opplastingsdemo. Gjennomsnittlig ECE var 0,316, og en fast grense på 0,5 ga lav presisjon for flere funn. Modellen skal derfor presenteres som et avsluttet forskningsprosjekt med dokumenterte testresultater, ikke som et verktøy for pasienter eller medisinske beslutninger.

Filer fra siste kontroll:

- `audit_results/summary.json` har samlet resultat og kjøretid.
- `audit_results/per_disease_metrics.csv` har resultat for hver av de 14 klassene.
- `audit_results/test_predictions.npz` inneholder modellens testprediksjoner for videre analyse.
