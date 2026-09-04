
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


# 3. assertion을 exception handling으로 사용

def get_user_age():
    try:
        age = int(input("나이를 입력하세요: "))
        # ❌ 잘못된 사용: 입력값 검증을 assert로 수행
        assert age >= 0, "나이는 음수가 될 수 없습니다."
        return age
    except AssertionError as e:
        print(f"입력 오류: {e}")
        return 0


def process_payment(amount, balance):
    try:
        # ❌ 잘못된 사용: 잔액 부족 체크를 assert와 try-except로 제어
        assert amount <= balance, "잔액이 부족합니다."
        balance -= amount
        return balance, "결제 성공"
    except AssertionError:
        return balance, "결제 실패"


def fetch_from_database(user_id: str):
    try:
        print(user_id)
    except Exception as e:
        print(e)


def get_active_user_profile(user_id):
    db_response = fetch_from_database(user_id)  # 유저 정보 조회 (딕셔너리 또는 None)

    try:
        # ❌ 잘못된 사용: 외부 데이터 존재 여부 확인을 assert로 처리
        assert db_response is not None, "존재하지 않는 사용자"
        return db_response['profile']
    except AssertionError:
        # 유저가 없을 때 기본 프로필을 반환하는 흐름 제어
        return {"name": "Guest", "role": "anonymous"}


