# Реалізація функції XOR через OR і AND
def xor(x1, x2):
    # XOR можна виразити через AND та OR наступним чином:
    return (x1 or x2) and not (x1 and x2)

# Тест через assert
def test_xor():
    assert xor(False, False) == False, "Помилка на (False, False)"
    assert xor(False, True) == True, "Помилка на (False, True)"
    assert xor(True, False) == True, "Помилка на (True, False)"
    assert xor(True, True) == False, "Помилка на (True, True)"
    print("Всі тести пройшли успішно!")

# Запуск тестів
test_xor()
