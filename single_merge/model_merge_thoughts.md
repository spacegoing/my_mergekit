## motivation

* **存在的问题：** 在 RL 阶段，同时训练 math, code, stem, rl 领域的数据，会导致不同领域的能力的竞争，模型无法同时学到最优的 math, code, stem, align 能力。
* **解决方案：** 一种可能的解决思路是，先分别在不同的领域上训练单独的领域专家模型，然后再 merge 成一个统一的模型。我们希望这个统一的模型可以保持每个领域专家模型的能力，在所有的任务上。
* **好处：**
* a. **in-domain:** 保证了不同领域内的训练数据是 multi-task 协同提升的
* b. **cross-domain:** 避免了不同领域间的数据冲突；

## terms

*(此处对应图 1：Separate Models, Ensemble Learning, Model Merging 的对比示意图)*

> **Figure 1:** An illustration of the ensemble learning paradigm versus the model merging paradigm. (a)  separate models for  tasks, (b) Assemble  separate models for  tasks, (c) A merged model for  tasks.

1. **定义：** 假设从一个 base 模型  出发，在不同领域的数据集 A, B, C 上进行训练（SFT 或者 RL），得到多个模型 ，通过 model merge 的方法，可以获得一个新的模型 merged model，记作 。我们希望该模型可以同时在领域 A, B, C 上表现良好，甚至优于原来的 domain specific LLMs；
2. **basic notations:**
* a. **delta parameters (task vector):** ，即，从基础模型出发，训练后所带来的参数变化量；
* b. **parameters interference:** 不同的 task vector 在 1. direction, 2. magnitude 上存在冲突，直接对这些 task vector 进行相加和可能会导致各自的效果都会削弱；
* c. **parameter redundancy:** task vector (or delta parameters) 是高度冗余的；丢弃 90% 左右的参数，几乎不会对模型的性能产生什么太大的影响；（**该结论在 small-scale models 上是成立的，但是在我们的 300b+ 模型不成立**）
* d. **orthogonality of task vectors:** 不同的 task vectors 之间的相似度极低，几乎是完全正交的；（**该结论在 large-scale LLMs 上也不成立，不同的 task vectors 之间是存在协同性的**）


3. **basic methods**
* a. **linear interpolation:** 直接对  进行操作，可分为 1. 不加权：simple averaging 2. 加权：fisher averaging；




* b. **task arithmetic:** 在  的基础上，对各自的 task vector 操作，lambda 一般为 0.3~0.5 之间



*(此处对应 Task vector 原理图：(a) Task vector, (b) Learning via addition, (c) Forgetting via negation, (d) Task analogies)*



---

## 现有方法的表现

1. **task arithmetic，可以作为 model merge 的 baseline；**
* a. **结论：** 调整不同 task vector 的 alpha 权重，可以改变模型对不同任务的倾向性，但不会对整体性能有明显的提升；因此默认假设所有的任务都是同等重要的，在此基础上对 model merge 的算法做改进；


2. **TIES-Merging**
* a. **TIES 的效果明显低于 baseline（平均低 2 个点左右），说明其在 large scale LLMs 上不 work；**


3. **DARE**
* a. **DARE 的结果和 baseline 差距不大，甚至略微低于 baseline；**



### 为什么？

TIES 和 DARE 的假设前提都是，模型的参数空间是高度稀疏化的，所以，分别通过 magnitude-based trim 以及 random dropout，先对 task vector 进行稀疏化处理，然后再进行 merge；好处是：

* a. 去掉了参数的冗余性；
* b. 部分缓解了模型的参数冲突；

**但问题是，对小模型的参数空间的假设，在大模型上真的成立吗？**
我们发现并不成立，大模型的参数空间要比小模型更稠密，基于 magnitude 的 dropout，当丢弃的比例达到 70% 的时候，模型就已经崩溃了；但在小模型上，可以丢弃 90% 甚至 99% 的参数而不对模型的性能产生太大的影响；

---

## 解决方案

根据以上结论，显然，从直觉上来讲，可以得到一种解决思路：**尽可能保证模型的稠密性，只在必要的时候，对参数进行最小的稀疏化，来减少 parameter interference；**

```
# Algorithm 1: Conflict-Aware Density-Preserving Model Merging

# Require: 
#   Base model: theta_0
#   Specialist models: {theta_i}, i=1 to K
#   Coefficients: {alpha_i}, i=1 to K
#   Threshold: tau

def conflict_aware_density_preserving_merge(theta_0, specialists, alphas, tau):
    K = len(specialists)
    d = get_parameter_dimension(theta_0) # 参数总维度
    
    # Step 1: Compute task vectors
    # tau_i = theta_i - theta_0 for i = 1, ..., K
    task_vectors = [theta_i - theta_0 for theta_i in specialists]
    
    # Step 2: Rank each task vector by magnitude and compute importance
    # importance_i[j] = d + 1 - rank_i[j]
    importances = []
    for tv in task_vectors:
        # 对参数绝对值大小进行排序（由大到小）
        ranks = compute_magnitude_ranks(tv) 
        importances.append(d + 1 - ranks)
        
    # Step 3: Initialize masks m_i = 1 for all i
    masks = [ones_like(theta_0) for _ in range(K)]
    
    # 遍历每一个参数位置 j
    for j in range(d):
        # 检查在该位置是否存在符号（方向）冲突
        if has_sign_conflict([tv[j] for tv in task_vectors]):
            # 分组：正向组 P 和 负向组 N
            P_indices = [i for i in range(K) if task_vectors[i][j] > 0]
            N_indices = [i for i in range(K) if task_vectors[i][j] < 0]
            
            # 计算各组的重要性总和
            I_P = sum([importances[i][j] for i in P_indices])
            I_N = sum([importances[i][j] for i in N_indices])
            
            # 冲突处理逻辑
            if abs(I_P - I_N) > tau:
                # 丢弃重要性较低的一组 (Drop less important group)
                drop_indices = P_indices if I_P < I_N else N_indices
            else:
                # 随机选择一组丢弃 (Random drop)
                drop_indices = random_select([P_indices, N_indices])
            
            # 将被丢弃组的 mask 设为 0
            for i in drop_indices:
                masks[i][j] = 0
                
    # Step 4: Apply masks and rescale
    # p_i 是丢弃率，用于后续恢复参数期望值
    processed_task_vectors = []
    for i in range(K):
        tilde_tau_i = masks[i] * task_vectors[i]
        p_i = 1 - (count_nonzero(masks[i]) / d)
        # 防止除以 0，进行缩放
        hat_tau_i = tilde_tau_i / (1 - p_i)
        processed_task_vectors.append(hat_tau_i)
        
    # Step 5: Aggregate
    # theta_merged = theta_0 + sum(alpha_i * hat_tau_i)
    theta_merged = theta_0 + sum([alphas[i] * processed_task_vectors[i] for i in range(K)])
    
    return theta_merged
```


**计算步骤（补充说明）：**

1. 计算各自的 task vector 内部的 importance ranking，作为后续对比的指标。不使用原始的幅度值的原因：避免不同的任务幅度值差异过大，某一类任务直接被掩盖掉；
2. 按照参数的方向进行分组，如果两组间的重要性差距过大，丢弃不重要的一组；如果接近，则随机选择一组保留；
3. 对处理后的参数进行 rescale 来恢复原始的期望值，最后再加上 base 模型。

如果您需要将其转换成具体的 PyTorch 逻辑代码，我可以继续为您实现。

**计算步骤：**

1. 计算各自的 task vector 内部的 importance ranking，作为后续对比的指标。不使用原始的幅度值的原因：避免不同的任务幅度值差异过大，某一类任务直接被掩盖掉；
2. 按照参数的方向进行分组，如果两组间的重要性差距过大，丢弃不重要的一组；如果接近，则随机选择一组保留；
3. 对处理后的参数进行 rescale 来恢复原始的期望值，最后再加上 base 模型；

---

## 效果

可以看到，我们的方法超过了 task arithmetic, DARE, TIES;

1. 当只 merge 一个 base 模型 + 两个 RL 模型 (math, code) 的时候，dots 2.0 就可以超过其他的模型；merged 之后的模型的数学能力保持不变，代码能力和 general alignment 能力甚至有提升；
* a. **解释：**
* i. 可以对 RL 训练得到的模型的 checkpoint 进行适度的线性外推来模拟训练更多步数的结果：
* ii. 大模型的参数稠密性使得不同任务间存在协同性：类似于向量加法，和向量的投影对每个任务的子向量有促进作用。





**分析：在 SFT 模型通过 RL 训练得到 domain-specific models 的过程中，变化最大的参数是什么？**
SFT -> math RL 以及 SFT -> code RL 模型的过程中，变化最大的是前几层的 **moe gate layer**；参数变化的数量级远大于其他层；但是 math 和 code 模型之间，moe gate layer 的区别并不大，说明 SFT -> Math RL 以及 SFT -> code RL 过程中，moe gate layer 参数变化的趋势是一致的；
**结论：可以联合 merge math 模型和 code 模型，不会产生太大的 parameter interference;**

---

## general performance

4. **目前的主要结论**
* i. **In Domain Performance:** 效果一般差于 domain-specific models，也可能差于 multi-task learning 的效果；但可以通过优化超参数的方法，获得更好的表现；比如添加 DARE 之后，可能获得  的结果：WizardLM-13B + WizardMath-13B -> better instruction following + better mathematical reasoning results;
* ii. 随着 domain-specific models 数量的上升，merged model 的 performance 存在下降；
* iii. 通常情况下，merged model 的 OOD 效果还不错，有比较 robust 的 ood generalization 能力；
* iv. 在训练目标、数据冲突严重的领域，一般情况下，merge model > mix data


5. **对于 RL 阶段的帮助：**
* i. merged model 的能力可能比 multitask joint learning & RFT 的结果要好；
* ii. **显然，Merged model 的熵很高，在此基础上进行 RL，有助于通过增加 exploration 的能力，来获得更好的 performance upperbound;**
* 1. 坏处是，merged model 的熵很高，会导致输出不稳定，需要通过 RL 过程来降低熵，稳定模型；






6. **在 llm 上的应用**
* a. **pretrain 阶段**
* i. 在 pretrain 阶段，可以对同一个训练轨迹上的不同 checkpoint 进行 average，来提高 pretrain 阶段的训练稳定性；


* b. **RL 阶段**
* i. 缓解不同领域间的竞争关系；


* c. **instruction -> thinking 阶段**
* i. 应用最广泛且效果得到验证的是，将 model merge 应用在 long-cot model + instruction model 上，有不少工作发现，通过这种，可以直接让模型学会 “adaptive thinking”，在减少大量 thinking tokens 的同时，取得基本和 long-cot 模型一样的推理效果；
* 1. 对我们工作的影响：可以作为后续训练 adaptive thinking 模型的起点，使用 model merge 作为训练的起点，在此基础上，再继续进行后续的训练；
