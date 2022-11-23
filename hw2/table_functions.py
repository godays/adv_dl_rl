from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from itertools import islice, chain
from typing import Iterable, Any, Tuple, Optional, List, Set


class TableDistribution:
    """Итеративно обновляет вероятности значений value"""

    def __init__(self, iterable: Iterable[Any] = None):
        if iterable is not None:
            self.counter = Counter(iterable)
            self.size = sum(self.counter.values())
        else:
            self.counter = Counter()
            self.size = 0

    def update(self, value: Any):
        self.size += 1
        self.counter[value] += 1

    def get_proba(self, value: Any) -> float:
        return self.counter[value] / self.size

    def __getitem__(self, key: Any) -> float:
        return self.get_proba(key)

    def keys(self) -> Iterable[Any]:
        return self.counter.keys()

    def show(self, top: Optional[int] = None) -> str:
        representation = f'TableDistribution(total={self.size}, unique={len(self.counter)})'
        if top is not None:
            representation += f' [top {top}]'
        pairs = islice(sorted(self.counter.items(), key=lambda k: k[1], reverse=True), top)
        return representation + '\n' + '\n'.join(f'\t{val}\t{count / self.size:.6f}\tn={count}' for val, count in pairs)

    def __repr__(self) -> str:
        return self.show()


def _get_union_keys(functions: List[TableFunction]) -> Set[Any]:
    return set(chain.from_iterable(f.keys() for f in functions))


class TableFunction(ABC):
    """Родительский класс для итеративно обновляемых табличных функций"""

    @abstractmethod
    def update(self, key: Any, value: float) -> None:
        pass

    @abstractmethod
    def merge(self, functions: List[TableFunction]) -> None:
        """Обновляет значения текущей функции на основе значений каждой из functions.
        Предполагается, что каждая из functions получениа из текущей путём многократного
        вызова update. Используется для объединения результатов параллельных обновлений
        в нескольких потоках"""
        pass

    @abstractmethod
    def get(self, key: Any) -> float:
        pass

    def __getitem__(self, key: Any) -> float:
        return self.get(key)

    @abstractmethod
    def items(self) -> Iterable[Tuple[Any, float]]:
        pass

    @abstractmethod
    def keys(self) -> Iterable[Any]:
        pass

    @abstractmethod
    def show(self, top: Optional[int] = None) -> str:
        pass

    def __repr__(self):
        return self.show()


class RunningMeans(TableFunction):
    """Итеративно обновляет средние (арифметические) значения value для каждого key"""

    def __init__(self):
        self.means = defaultdict(float)
        self.counts = defaultdict(int)
        self.total = 0

    def update(self, key: Any, value: float):
        m = self.means[key]
        n = self.counts[key]
        self.means[key] = (m * n + value) / (n + 1)
        self.counts[key] += 1
        self.total += 1

    def merge(self, functions: List[RunningMeans]) -> None:
        """Обновляет средние значения по каждому ключу на основе средних значений каждой из functions.
        Предполагается, что каждая из functions получениа из текущей путём многократного
        вызова update. Используется для объединения результатов параллельных обновлений
        в нескольких потоках"""
        assert len(functions) > 1  # Если нужно обновление по одной новой функции, то достаточно её и взять
        for key in _get_union_keys(functions):
            current_count = self.counts[key]
            current_mean = self.get(key)
            current_sum = current_mean * current_count
            new_sum = current_sum
            new_count = current_count
            for f in functions:
                f_count = f.counts[key]
                if f_count == current_count:
                    continue
                f_mean = f.get(key)
                f_sum = f_mean * f_count
                f_sum_delta = f_sum - current_sum
                new_sum += f_sum_delta
                f_count_delta = f_count - current_count
                new_count += f_count_delta
                self.total += f_count_delta
            self.means[key] = new_sum / new_count
            self.counts[key] = new_count

    def get(self, key: Any) -> float:
        #  Из defaultdict можно брать значения по несуществующему ключу,
        #  но тогда по этому ключу будет создано новое значение
        return self.means[key] if key in self.means else 0

    def items(self) -> Iterable[Tuple[Any, float]]:
        return self.means.items()

    def keys(self) -> Iterable[Any]:
        return self.means.keys()

    def show(self, top: Optional[int] = None) -> str:
        representation = f'RunningMeans(total={self.total}, unique={len(self.means)}'
        if top is not None:
            representation += f' [top {top}]'
        pairs = islice(sorted(self.items(), key=lambda k: k[1], reverse=True), top)
        return representation + '\n' + '\n'.join(f'\t{key}\t{value:.6f}\tn={self.counts[key]})'
                                                 for key, value in pairs)


class RunningExp(TableFunction):
    """Выполняет итеративное экспоненциальное сглаживание значений value для каждого key"""

    def __init__(self, alpha: float, merge_strategy: str = 'delta_weighted_mean'):
        assert 0 < alpha < 1
        assert merge_strategy in ['sequential',
                                  'mean', 'weighted_mean',
                                  'delta_mean', 'delta_weighted_mean']
        self.alpha = alpha
        self.merge_strategy = merge_strategy
        self.values = defaultdict(float)
        self.counts = defaultdict(int)
        self.total = 0

    def update(self, key: Any, value: float):
        if self.counts[key] > 0:
            self.values[key] = (1 - self.alpha) * self.values[key] + self.alpha * value
        else:
            self.values[key] = value
        self.counts[key] += 1
        self.total += 1

    def _merge_sequential(self, functions: List[RunningExp]) -> None:
        """Если функции functions[f1,f2,...,fN] были получены из текущей путём вызова update с последовательностями
        значений [X^1_1, X^1_2, ... X^1_{n_1}], ..., [X^N_1, X^N_2, ..., X^N_{n_N}], то обновляет значения текущей
        функции таким образом, чтобы результат был эквивалентен обновлению текущей функции по последовательности
        значений [X^1_1, X^1_2, ... X^1_{n_1}, ..., X^N_1, X^N_2, ..., X^N_{n_N}]"""
        for key in _get_union_keys(functions):
            current_count = self.counts[key]
            current_value = self.get(key)
            #  Сначала делаем обновление по первой функции:
            new_value = functions[0].get(key)
            new_count = functions[0].counts[key]
            for f in functions[1:]:
                f_count = f.counts[key]
                if f_count == current_count:
                    continue
                f_value = f.get(key)
                f_count_delta = f_count - current_count
                new_count += f_count_delta
                new_value = (new_value - current_value) * (1 - self.alpha) ** f_count_delta + f_value
            self.values[key] = new_value
            self.counts[key] = new_count

    def _merge_mean(self, functions: List[RunningExp]) -> None:
        """Усредняет значения тех функций, где были какие-то обновления"""
        for key in _get_union_keys(functions):
            current_count = self.counts[key]
            updated_functions = [f for f in functions if f.counts[key] > current_count]
            if len(updated_functions) == 0:
                continue
            new_count = self.counts[key]
            sum_values = 0
            for f in updated_functions:
                f_count = f.counts[key]
                f_count_delta = f_count - current_count
                f_value = f.get(key)
                sum_values += f_value
                new_count += f_count_delta
            self.values[key] = sum_values / len(updated_functions)
            self.counts[key] = new_count

    def _merge_weighted_mean(self, functions: List[RunningExp]) -> None:
        """Усредняет значения тех функций, где были какие-то обновления, с весами,
        равными числу обновлений"""
        for key in _get_union_keys(functions):
            current_count = self.counts[key]
            updated_functions = [f for f in functions if f.counts[key] > current_count]
            if len(updated_functions) == 0:
                continue
            weighted_sum_values = 0
            sum_weights = 0
            for f in updated_functions:
                f_count = f.counts[key]
                f_count_delta = f_count - current_count
                f_value = f.get(key)
                weighted_sum_values += f_count_delta * f_value
                sum_weights += f_count_delta
            self.values[key] = weighted_sum_values / sum_weights
            self.counts[key] = self.counts[key] + sum_weights

    def _merge_delta_mean(self, functions: List[RunningExp]) -> None:
        """Усредняет _приращения_ значений (относительно текущих) тех функций, где были какие-то обновления"""
        for key in _get_union_keys(functions):
            current_count = self.counts[key]
            current_value = self.get(key)
            updated_functions = [f for f in functions if f.counts[key] > current_count]
            if len(updated_functions) == 0:
                continue
            new_count = self.counts[key]
            sum_delta_values = 0
            for f in updated_functions:
                f_count = f.counts[key]
                f_count_delta = f_count - current_count
                f_value = f.get(key)
                sum_delta_values += f_value - current_value
                new_count += f_count_delta
            self.values[key] = current_value + sum_delta_values / len(updated_functions)
            self.counts[key] = new_count

    def _merge_delta_weighted_mean(self, functions: List[RunningExp]) -> None:
        """Усредняет _приращения_ значения тех функций, где были какие-то обновления, с весами,
        равными числу обновлений"""
        for key in _get_union_keys(functions):
            current_count = self.counts[key]
            current_value = self.get(key)
            updated_functions = [f for f in functions if f.counts[key] > current_count]
            if len(updated_functions) == 0:
                continue
            weighted_sum_delta_values = 0
            sum_weights = 0
            for f in updated_functions:
                f_count = f.counts[key]
                f_count_delta = f_count - current_count
                f_value = f.get(key)
                weighted_sum_delta_values += f_count_delta * (f_value - current_value)
                sum_weights += f_count_delta
            self.values[key] = current_value + weighted_sum_delta_values / sum_weights
            self.counts[key] = self.counts[key] + sum_weights

    def merge(self, functions: List[RunningExp]) -> None:
        """Обновляет сглаженные значения по каждому ключу на основе сглаженных значений каждой из functions.
        Предполагается, что каждая из functions получениа из текущей путём многократного
        вызова update. Используется для объединения результатов параллельных обновлений
        в нескольких потоках"""
        assert len(functions) > 1  # Если нужно обновление по одной новой функции, то достаточно её и взять
        assert all(f.alpha == self.alpha for f in functions)

        if self.merge_strategy == 'sequential':
            self._merge_sequential(functions)
        elif self.merge_strategy == 'mean':
            self._merge_mean(functions)
        elif self.merge_strategy == 'weighted_mean':
            self._merge_weighted_mean(functions)
        elif self.merge_strategy == 'delta_mean':
            self._merge_delta_mean(functions)
        elif self.merge_strategy == 'delta_weighted_mean':
            self._merge_delta_weighted_mean(functions)
        else:
            raise NotImplementedError

        new_total = self.total
        for f in functions:
            new_total += f.total - self.total
        self.total = new_total

    def get(self, key: Any) -> float:
        #  Из defaultdict можно брать значения по несуществующему ключу,
        #  но тогда по этому ключу будет создано новое значение
        return self.values[key] if key in self.values else 0

    def items(self) -> Iterable[Tuple[Any, float]]:
        return self.values.items()

    def keys(self) -> Iterable[Any]:
        return self.values.keys()

    def show(self, top: Optional[int] = None) -> str:
        representation = f'RunningExp(alpha={self.alpha}, total={self.total}, unique={len(self.values)}'
        if top is not None:
            representation += f' [top {top}]'
        pairs = islice(sorted(self.items(), key=lambda k: k[1], reverse=True), top)
        return representation + '\n' + '\n'.join(f'\t{key}\t{value:.6f}\tn={self.counts[key]})'
                                                 for key, value in pairs)
