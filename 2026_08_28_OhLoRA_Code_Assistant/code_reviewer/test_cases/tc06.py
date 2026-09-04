
# 1. 동일한 인수가 반복되는 코드

# 주문 정보를 처리하는 개별 함수들
def calculate_total_price(item_name, price, quantity, address):
    # 총액 계산 (수량별 할인 등 적용 가능)
    base_price = price * quantity
    print(f"[{item_name}] {quantity}개 기본 금액: {base_price}원")
    return base_price

def calculate_shipping_fee(item_name, price, quantity, address):
    # 배송지에 따른 배송비 계산
    if "제주" in address:
        shipping_fee = 5000
    else:
        shipping_fee = 3000
    print(f"[{address}] 배송비: {shipping_fee}원")
    return shipping_fee

def print_order_summary(item_name, price, quantity, address):
    # 주문 요약 정보 출력
    total = calculate_total_price(item_name, price, quantity, address)
    shipping = calculate_shipping_fee(item_name, price, quantity, address)
    print(f"--- 주문서 ---")
    print(f"상품명: {item_name} / 최종 결제 금액: {total + shipping}원")

# 함수 호출 (매번 4개의 인수를 똑같이 넘겨야 함)
item_name = "맥북 에어"
price = 1200000
quantity = 1
address = "제주도 제주시"

print_order_summary(item_name, price, quantity, address)

# 2. 상태 값으로 판단되는 값을 조건으로 하여 body가 1줄 정도의 일정한 패턴인 if-elif-else 처리가 있는 경우


class Order:
    def __init__(self):
        self.state = "PENDING"

    def process(self):
        if self.state == "PENDING":
            print("결제를 대기 중입니다.")
        elif self.state == "PAID":
            print("대금을 확인하여 배송을 준비합니다.")
        elif self.state == "SHIPPED":
            print("이미 배송된 상품입니다.")
        elif self.state == "CANCELLED":
            print("취소된 주문입니다.")


class DiscountCalculator:
    def calculate(self, grade, price):
        if grade == "VIP":
            return price * 0.8
        elif grade == "GOLD":
            return price * 0.9
        elif grade == "SILVER":
            return price * 0.95
        else:
            return price


class NotificationService:
    def send_notification(self, type, message):
        if type == "EMAIL":
            print(f"이메일로 발송: {message}")
        elif type == "SMS":
            print(f"SMS문자로 발송: {message}")
        elif type == "PUSH":
            print(f"앱 푸시로 발송: {message}")


def send_notification(order_status):
    if order_status == "PAYMENT_PENDING":
        print("결제 대기 안내 문자 발송")
    elif order_status == "SHIPPING":
        print("배송 시작 알림톡 발송")
    elif order_status == "DELIVERED":
        print("배송 완료 알림 발송")
    else:
        print("기본 고객 센터 안내 발송")


def calculate_discount(payment_method, price):
    if payment_method == "CREDIT_CARD":
        return price * 0.05
    elif payment_method == "NAVER_PAY":
        return price * 0.08
    elif payment_method == "COUPON":
        return price * 0.10
    else:
        return 0


def check_permission(user_grade):
    if user_grade == "BRONZE":
        return ["VIEW_POST"]
    elif user_grade == "SILVER":
        return ["VIEW_POST", "WRITE_POST"]
    elif user_grade == "GOLD":
        return ["VIEW_POST", "WRITE_POST", "VIP_LOUNGE"]
    else:
        return []
