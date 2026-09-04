
# 1. exception ignored

try:
    # 오류가 날 수 있는 코드
    num = int(input("숫자 입력: "))
except:
    # 예외를 삼켜버리는 경우 (에러를 무시함)
    pass

try:
    num = int(input("숫자 입력: "))
except Exception:
    # 예외를 삼켜버리는 경우 (에러를 무시함)
    pass

try:
    num = int(input("숫자 입력: "))
except BaseException as exp:
    # 예외를 삼켜버리는 경우 (에러를 무시함)
    pass

try:
    # 오류가 날 수 있는 코드
    num = int(input("숫자 입력: "))
except:
    # exception occured 출력하는 경우
    print('exception occurred')

try:
    num = int(input("숫자 입력: "))
except IOError:
    pass

try:
    num = int(input("숫자 입력: "))
except IOError as ioe:
    pass

# 2. function-related errors


def func_test(a: int = 1, b: float = 2.0):
    print(a)
    print(b)


# 위반 예시
def create_user(name: str, data: dict = {}):
    data["name"] = name
    return data


# 위반 예시
def add_to_list(item: str, items: list = []):
    items.append(item)
    return items


# 위반 예시
def configure_settings(options: dict = {"theme": "light", "debug": False}):
    options["debug"] = True  # 특정 상황에서 디버그 모드를 켬
    return options


# 위반 예시
def create_user_1(
        name: str,
        data: dict = {}
):
    data["name"] = name
    return data


# 위반 예시
def add_to_list_1(
        item: str, items: list = []
):
    items.append(item)
    return items


# 위반 예시
def configure_settings_1(
        options: dict = {"theme": "light", "debug": False}
):
    options["debug"] = True  # 특정 상황에서 디버그 모드를 켬
    return options
