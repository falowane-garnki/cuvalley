# CuValley Hack 2022 - Falowane Garnki

#### Sztuczny analizator temperatury żużla wewnątrz Pieca Zawiesionowego Huty Miedzi Głogów I

## Korzystanie z modelu
Cały kod potrzebny do przeprowadzenia predykcji znajduje się w folderze `src/load_and_predict`, w jupyter notebook'u `load_and_predict.ipynb`.<br>
Aby móc go prawidłowo uruchomić należy umieścić odszyfrowane dane testowe w plikach .gz w folderze `src/data`.

Komórki w notebooku należy uruchamiać po kolei. Dane zostaną rozpakowane,
połączone w jeden duży plik .csv, a następnie zostanie wstępne
przetwarzania danych, które trafią do wyszkolonego wcześniej modelu

## Wymagania
- środowisko Python3
- zainstalowane biblioteki wymienione w requirements.txt. Aby zainstalować: 
```
pip install -r requirements.txt
```



## Opis pozostałych plików
- folder *linear regression* zawiera notebooki w których testowaliśmy kolejne modele, może nie działać prawidłowo ze względu na zmiany w ścieżkach
 
# Autorzy
Wojciech Jasiński

Kacper Kłusek

Adam Kwiatkowski

Weronika Witek

Przemysław Węglik