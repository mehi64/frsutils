اضافه کردن CuPy به کل پروژه کار درستی نیست؛ ولی اضافه کردنش به‌عنوان backend اختیاری برای هسته‌ی محاسباتی FRsutils خیلی ارزشمند است.
کوپای CuPy را به‌صورت اختیاری و محدود اضافه کن، نه به تمام پروژه. backend="numpy" | "cupy" | "auto"

حذف FRSMOTE و keel audit از frsutils
آماده کردن برای ۲ مقاله


--------------------------------------------------------------
۱. اول . CuPy را اضافه کن، ولی نه مستقیم؛ با Backend Layer اضافه کن
۷. Array API مسیر آینده باشد، CuPy فقط اولین backend باشد
۲. مهم‌ترین قابلیت فنی: blockwise similarity matrix
۳. بعد از حذف FRSMOTE، حتماً چند estimator آماده بده
FuzzyRoughPositiveRegionScorer,
FuzzyRoughDependencyScorer
FuzzyRoughFeatureSelector

۵. FRsutils را با scikit-learn سازگار نگه دار، حتی بدون FRSMOTE
۶. یک Benchmark Suite رسمی بساز
. یک explain() یا diagnose() برای fuzzy-rough results بده

۹. Config recipes آماده بساز
**************************************************************************************************************
approximate similarity for making library faster
**************************************************************************************************************
| اولویت | قابلیت                                                           | ارزش برای پذیرش |     زمان/ریسک | نظر نهایی                                |
| -----: | ---------------------------------------------------------------- | --------------: | ------------: | ---------------------------------------- |
|      1 | **Similarity Matrix Engine تمیز و سریع**                         |       خیلی زیاد |         متوسط | بهترین گزینه برای شروع                   |
|      2 | **Public API برای lower/upper approximation و positive region**  |       خیلی زیاد |   کم تا متوسط | باید حتماً انجام شود                     |
|      3 | **Backend abstraction برای NumPy/CuPy، حتی اگر CuPy کامل نباشد** |            زیاد |         متوسط | بهتر از اضافه‌کردن خام CuPy              |
|      4 | **FuzzyRoughPositiveRegionScorer / DependencyScorer**            |            زیاد |         متوسط | library را کاربردی می‌کند                |
|      5 | **FuzzyRoughFeatureSelector ساده**                               |       خیلی زیاد | متوسط تا زیاد | اگر وقت داری، خیلی ارزشمند است           |
|      6 | **Benchmark suite کوچک CPU/GPU/competitor**                      |            زیاد |         متوسط | برای مقاله خیلی کمک می‌کند               |
|      7 | **FRSMOTE**                                                      |       خیلی زیاد |          زیاد | ارزش علمی بالا، ولی بهتر است جدا بماند   |
|      8 | **KEEL Audit**                                                   |   متوسط تا زیاد |    زیاد/آشفته | برای paper اصلی FRsutils اولویت پایین‌تر |
|      9 | **CuPy کامل برای کل library**                                    |   متوسط تا زیاد |          زیاد | فعلاً نه؛ فقط selective                  |
|     10 | **Rule induction / reduct کامل**                                 |            زیاد |     خیلی زیاد | فعلاً انجام نده                          |



------------------------------------------------------------------------------------------------------------------------------------
۱. Similarity Matrix Engine — اولویت اول

این مهم‌ترین چیزی است که پیشنهاد می‌کنم.

الان خود پروژه already یک build_similarity_matrix(X, config=None, **kwargs) در similarities.py دارد و این تابع پل بین public params و محاسبات داخلی معرفی شده است. پس این قابلیت از صفر نیست؛ باید آن را تبدیل کنی به یک feature رسمی و قوی.

چرا ارزشش بالاست؟

در fuzzy-rough setها، تقریباً همه‌چیز به similarity relation وابسته است. اگر similarity matrix خوب، سریع، قابل تنظیم و قابل benchmark داشته باشی، FRsutils حتی بدون FRSMOTE هم ارزشمند می‌شود.

رقبا ممکن است الگوریتم‌های بیشتری داشته باشند، ولی تو می‌توانی بگویی:

FRsutils provides a reusable, configurable, and optionally accelerated similarity-matrix engine for fuzzy-rough computation.

این claim خوب است، چون هم core است، هم قابل استفاده توسط بقیه، هم بعداً FRSMOTE/feature selection/positive region روی آن ساخته می‌شوند.

دقیقاً چه اضافه کنی؟

حداقل این‌ها:

build_similarity_matrix(
    X,
    similarity="gaussian",
    similarity_sigma=0.5,
    backend="numpy",
    chunk_size=None,
    dtype="float64",
)

و اگر وقت داشتی:

iter_similarity_blocks(
    X,
    Y=None,
    similarity="gaussian",
    chunk_size=2048,
    backend="numpy",
)
چرا blockwise مهم است؟

چون similarity matrix برای n نمونه، n × n است. برای دیتاست بزرگ، همین نقطه شکست است. اگر FRsutils بتواند blockwise حساب کند، حتی بدون GPU هم حرف جدی دارد.

ارزش پذیرش

خیلی بالا. چون این قابلیت:

core fuzzy-rough است؛
به همه مدل‌ها کمک می‌کند؛
قابل benchmark است؛
تغییر زیادی در فلسفه پروژه نمی‌دهد؛
بعداً CuPy را طبیعی می‌کند.
۲. Public API برای approximation و positive region — اولویت دوم

الان FRsutils componentهای اصلی دارد: similarities، t-norms، implicators، OWA weights، fuzzy quantifiers و مدل‌های ITFRS/OWAFRS/VQRS. ولی اگر کاربر نتواند راحت بگوید «positive region را بده»، library خیلی کم‌استفاده می‌شود.

چه اضافه کنی؟

سه تابع public:

compute_lower_approximation(X, y, ...)
compute_upper_approximation(X, y, ...)
compute_positive_region(X, y, ...)

یا یک facade:

from frsutils import FuzzyRoughApproximator

approximator = FuzzyRoughApproximator(
    model="itfrs",
    similarity="gaussian",
)

result = approximator.fit(X, y).evaluate(X)

result.lower_approximation
result.upper_approximation
result.positive_region
result.boundary_region
چرا مهم است؟

چون بدون این API، FRsutils برای کاربر نهایی شبیه جعبه ابزار داخلی است. با این API، می‌شود library کاربردی.

ارزش پذیرش

خیلی بالا، حتی بالاتر از CuPy. چون داور یا کاربر می‌فهمد با library چه کاری می‌تواند انجام دهد.

۳. Backend abstraction برای NumPy/CuPy — اولویت سوم

ایده CuPy خوب است، ولی من نمی‌گویم فوراً کل library را GPU کنی. پیشنهاد دقیق‌تر:

اول backend abstraction بساز؛ بعد فقط similarity matrix را CuPy-ready کن.

یعنی:

build_similarity_matrix(..., backend="numpy")
build_similarity_matrix(..., backend="cupy")
build_similarity_matrix(..., backend="auto")
چرا نه CuPy کامل؟

چون اگر بگویی کل FRsutils GPU-ready است، باید همه‌چیز را تست و benchmark کنی. این ریسک بالاست.

اما اگر بگویی:

GPU acceleration is currently supported for similarity matrix construction.

این قابل دفاع است.

ارزش پذیرش

زیاد. چون GPU به library جذابیت می‌دهد، ولی فقط وقتی که:

optional dependency باشد؛
اگر CuPy نصب نیست، NumPy کار کند؛
خروجی CPU/GPU از نظر عددی تست شود؛
benchmark کوچک داشته باشی.
پیاده‌سازی کم‌ریسک
def build_array_module(backend="numpy"):
    if backend == "numpy":
        import numpy as xp
        return xp
    if backend == "cupy":
        import cupy as xp
        return xp
    if backend == "auto":
        try:
            import cupy as xp
            return xp
        except ImportError:
            import numpy as xp
            return xp

اما این را مستقیم همه‌جا پخش نکن. یک فایل بگذار:

FRsutils/core/backends.py

یا:

FRsutils/utils/backends.py


۴. PositiveRegionScorer / DependencyScorer — اولویت چهارم

این به‌نظرم بهترین جایگزین FRSMOTE داخل FRsutils است.

چرا؟ چون FRSMOTE از positive region استفاده می‌کند، ولی تو می‌خواهی FRSMOTE را جدا کنی. پس بهتر است positive region scoring را در FRsutils نگه داری و FRSMOTE بعداً از آن استفاده کند.

API پیشنهادی
from frsutils.scoring import FuzzyRoughPositiveRegionScorer

scorer = FuzzyRoughPositiveRegionScorer(
    model="itfrs",
    similarity="gaussian",
)

scores = scorer.fit_transform(X, y)

یا:

result = scorer.fit(X, y).score_samples(X)
خروجی‌ها
positive region score برای هر sample
boundary score
uncertainty score
class-wise lower approximation
class-wise upper approximation
ارزش رقابتی

زیاد. چون این library را از «چند تابع ریاضی» تبدیل می‌کند به «ابزار تحلیل fuzzy-rough».

ارزش برای مقاله

زیاد. چون می‌توانی در paper بگویی:

FRsutils exposes positive-region and approximation-based scoring as reusable primitives for downstream fuzzy-rough algorithms.

این دقیقاً همان چیزی است که بعداً FRSMOTE، feature selection و instance selection می‌توانند استفاده کنند.

۵. FuzzyRoughFeatureSelector — اولویت پنجم

این از نظر رقابت با fuzzy-rough-learn، RoughSets و Weka خیلی مهم است. یکی از ضعف‌های FRsutils بدون FRSMOTE این است که feature selection ندارد. اگر یک selector ساده اضافه کنی، library خیلی جدی‌تر می‌شود.

API پیشنهادی
from frsutils.feature_selection import FuzzyRoughFeatureSelector

selector = FuzzyRoughFeatureSelector(
    n_features=10,
    scoring="dependency",
    model="itfrs",
    similarity="gaussian",
)

X_new = selector.fit_transform(X, y)
نسخه ساده کافی است

لازم نیست از اول QuickReduct کامل و بسیار بهینه بسازی. یک نسخه ساده ولی تمیز:

هر feature را جداگانه score کن؛
top-k را انتخاب کن؛
بعداً greedy forward selection اضافه کن.
ارزش پذیرش

خیلی زیاد، چون feature selection یک کاربرد روشن است. وقتی داور بپرسد «کاربر با FRsutils چه می‌کند؟» جواب داری:

می‌تواند fuzzy-rough feature scoring و feature selection انجام دهد.

ریسک

متوسط تا زیاد. چون باید دقت الگوریتمی، تست، و مقایسه داشته باشی. ولی هنوز از FRSMOTE کم‌ریسک‌تر است.

۶. Benchmark Suite کوچک — اولویت ششم

این feature الگوریتمی نیست، ولی برای پذیرش خیلی ارزش دارد.

چه benchmarkی؟

سه benchmark کافی است:

1. similarity matrix runtime:
   NumPy dense vs NumPy blockwise

2. optional GPU:
   NumPy vs CuPy برای build_similarity_matrix

3. model comparison:
   ITFRS vs OWAFRS vs VQRS روی چند dataset کوچک sklearn

خروجی:

benchmark_results/
    runtime.csv
    memory.csv
    correctness.csv
    report.md
چرا مهم است؟

چون اگر similarity matrix و CuPy اضافه کنی ولی benchmark نداشته باشی، claim ضعیف است. Benchmark نشان می‌دهد این قابلیت واقعی است.

ارزش پذیرش

زیاد. مخصوصاً برای software paper.

۷. FRSMOTE — ارزش بالا، ولی بهتر برای مقاله جدا

حالا برسیم به FRSMOTE.

FRSMOTE از نظر ارزش علمی احتمالاً از همه بالاتر است. چون یک الگوریتم مشخص و قابل benchmark است. طبق وضعیت فعلی پروژه، FRSMOTE concrete oversampler است، از fuzzy-rough model و similarity matrix استفاده می‌کند، positive region را برای انتخاب seed به‌کار می‌برد، و flat parameter interface دارد.

اما برای FRsutils به‌عنوان library paper دو مشکل دارد:

scope را سنگین می‌کند؛
بهتر است خودش مقاله جدا شود.
تصمیم پیشنهادی من

FRSMOTE را داخل FRsutils paper اصلی نکن، اما hookهای لازم را نگه دار:

compute_positive_region(...)
rank_samples_by_positive_region(...)
FuzzyRoughPositiveRegionScorer
build_similarity_matrix(...)

بعداً package یا paper جدا:

frsmote
depends on frsutils
ارزش پذیرش

برای FRSMOTE paper: خیلی زیاد.
برای FRsutils paper: زیاد، ولی پرریسک و scope-bloating.

پیشنهاد نهایی

فعلاً FRSMOTE را جدا کن، اما FRsutils را طوری بساز که FRSMOTE بدون کپی‌کردن کد بتواند روی آن سوار شود.

۸. KEEL Audit — ارزش متوسط تا زیاد، ولی نه برای هدف فعلی

KEEL Audit ایده خوبی است، اما در وضعیت فعلی ظاهراً خیلی experimental و چندنسخه‌ای است؛ فایل پروژه می‌گوید KEEL_Audit_Utility چند کپی working دارد و باید canonical شود. همچنین branch فعلی FRSMOTE و KEEL audit را با هم قاطی کرده است.

ارزشش کجاست؟

برای reproducible experiments عالی است:

check کردن dataset
split consistency
class imbalance
duplicate detection
train/test leakage
schema mismatch
اما مشکلش چیست؟

به core fuzzy-rough خیلی نزدیک نیست. اگر FRsutils را به‌عنوان core fuzzy-rough toolkit معرفی می‌کنی، KEEL Audit ممکن است focus را پخش کند.

تصمیم پیشنهادی

فعلاً حذف از paper اصلی. بعداً می‌تواند package یا module جدا شود:

frsutils-keel

یا:

frsutils[keel]
ارزش پذیرش

برای paper ابزار آزمایش: زیاد.
برای paper اصلی FRsutils core: متوسط.

۹. CuPy کامل برای همه‌چیز — فعلاً نه

اگر منظورت این باشد که کل ITFRS/OWAFRS/VQRS و همه componentها با CuPy کار کنند، من فعلاً پیشنهاد نمی‌کنم.

چرا؟

چون هزینه پنهان زیاد دارد:

همه توابع باید با NumPy و CuPy سازگار باشند؛
type conversion مشکل می‌شود؛
CI با GPU نداری؛
تست عددی سخت‌تر می‌شود؛
ممکن است سرعت فقط برای dataset بزرگ بهتر شود؛
داور اگر benchmark بخواهد، باید جواب داشته باشی.
راه بهتر

فقط این را انجام بده:

CuPy support for similarity matrix construction

و بعداً:

experimental backend support for approximation models
۱۰. Rule induction / reduct کامل — فعلاً انجام نده

این‌ها ارزش علمی دارند، ولی الان برای تو مناسب نیستند:

QuickReduct کامل
LEM2
rule induction
rough-set reduct search
decision rules

چون هم زمان‌برند، هم رقبا در این‌ها سابقه زیاد دارند، هم تو را از مسیر اصلی دور می‌کنند.

رتبه‌بندی نهایی با امتیاز

امتیازها از ۱۰ هستند.

قابلیت	ارزش علمی	ارزش نرم‌افزاری	تمایز از رقبا	سختی	پیشنهاد
Similarity Matrix Engine	8	10	8	5	انجام بده
Lower/Upper/Positive Region API	8	10	7	4	انجام بده
Backend abstraction	7	9	8	5	انجام بده
CuPy فقط برای similarity matrix	7	8	8	6	اگر وقت داری انجام بده
PositiveRegionScorer	8	9	8	5	انجام بده
DependencyScorer	8	8	7	5	انجام بده
FuzzyRoughFeatureSelector ساده	9	9	8	7	اگر وقت داری، خیلی خوب
Benchmark suite کوچک	7	9	7	5	انجام بده
FRSMOTE	10	8	10	8	برای paper جدا نگه دار
KEEL Audit	7	8	7	8	فعلاً نه، بعداً جدا
Full CuPy backend	7	8	8	9	فعلاً نه
Rule induction / reduct کامل	8	7	6	9	فعلاً نه
اگر فقط ۲ هفته وقت داری

این کارها را بکن:

Similarity Matrix Engine تمیز
blockwise similarity
public API برای positive region
PositiveRegionScorer
benchmark کوچک
README و example خوب

CuPy را فقط اگر خیلی سریع شد، به similarity matrix اضافه کن.

اگر ۱ ماه وقت داری

این ترکیب بهترین است:

Similarity Matrix Engine
blockwise computation
backend abstraction
CuPy optional برای similarity matrix
PositiveRegionScorer
DependencyScorer
FuzzyRoughFeatureSelector ساده
Benchmark suite
docs/examples/paper.md

این نسخه به‌نظرم واقعاً قابلیت معرفی دارد.

اگر ۲ ماه وقت داری

آن‌وقت می‌توانی یکی از این دو مسیر را اضافه کنی:

مسیر A — paper اصلی FRsutils را قوی‌تر کن

اضافه کن:

FuzzyRoughFeatureSelector بهتر
benchmark با fuzzy-rough-learn / RoughSets
چند dataset واقعی
CuPy benchmark
مسیر B — FRSMOTE را جداگانه آماده کن

نگه دار:

FRsutils = core fuzzy-rough toolkit
FRSMOTE = downstream algorithm using FRsutils

این مسیر از نظر استراتژیک بهتر است، چون بعداً دو paper داری، نه یکی.

پیشنهاد نهایی من

برای تو، با هدف «کمترین آشفتگی + بیشترین ارزش پذیرش»، ترتیب تصمیم این باشد:

Similarity Matrix Engine + blockwise computation
compute_lower/upper/positive_region public API
PositiveRegionScorer
Backend abstraction
CuPy فقط برای similarity matrix
DependencyScorer
FuzzyRoughFeatureSelector ساده
Benchmark suite
FRSMOTE جداگانه، بعد از تثبیت FRsutils
KEEL Audit جداگانه یا فعلاً حذف

اگر بخواهم خیلی خلاصه بگویم:

الان ارزشمندترین کاری که می‌توانی بکنی این نیست که FRsutils را پر از feature کنی؛ بلکه این است که آن را به یک هسته سریع و قابل اتکا برای fuzzy-rough computation تبدیل کنی. ستون‌های اصلی‌اش باید similarity matrix، approximation API، positive-region scoring، backend abstraction و benchmark باشند. FRSMOTE و KEEL Audit ارزش دارند، ولی بهتر است بعداً به‌عنوان extension یا مقاله جدا از روی همین هسته ساخته شوند.







