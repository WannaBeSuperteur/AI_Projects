Diagram 생성 알고리즘

## 실행 방법

```commandline
python test_draw_diagram.py
```

* 입력
  * ```diagram.txt``` - 다이어그램 포맷으로 작성된 텍스트 파일

```python
model = [
    [1, 500, 300, "circle", "dashed", (255, 0, 0), (0, 0, 0), [2, 3]],
    [2, 700, 300, "square", "dashed", (0, 255, 0), (0, 0, 0), [3, 4]],
    [3, 500, 500, "triangle", "dashed", (0, 0, 255), (0, 0, 0), [1, 2]],
    [4, 700, 500, "rectangle", "dashed", (0, 255, 255), (0, 0, 0), [2, 3]],
    [5, 500, 700, "pentagon", "dashed", (255, 255, 0), (0, 0, 0), [4, 5]],
    [6, 700, 700, "hexagon", "dashed", (255, 0, 255), (0, 0, 0), [5, 6]],
    [7, 500, 900, "octagon", "dashed", (0, 255, 255), (0, 0, 0), [6, 7]],
```

* 출력
  * ```diagram.png``` - 생성된 다이어그램 이미지