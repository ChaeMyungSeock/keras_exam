# 02. 자동 미분(Autograd)
import torch
#  2w^2+5 에 대해 미분

w = torch.tensor(2.0, requires_grad=True)
# 텐서에는 requires_grad라는 속성이 있습니다. 이것을 True로 설정하면 자동 미분 기능이 적용됩니다.


y = w**2
z = 2*y + 5

z.backward()

# .backward()를 호출하면 해당 수식의 w에 대한 기울기를 계산합니다.

print('수식을 w로 미분한 값 : {}'.format(w.grad))

