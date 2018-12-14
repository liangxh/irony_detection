# SemEval2019 Task3 数据观察

* 以下三个回合的对话以A1, B, A2表示

## 观察

* 若A，B出现两种情感，以A为准
* 若A出现两种情感，以后者为准
* 大部分样本可以仅靠一两个单词判断整段对话的情感
* 少部分样本的标注存在主观性（和个人理解不同）
* 少部分样本的情感依賴外面知识
* 有一些情感相对隐晦但对分辨能力的词组
* 具分辨力的单词，其负例的标签大部分是others
* 部分样本拼写很不规范

## Happy

* 若A，B出现两种情感，以A为准

```
A1: yes yay fun
B: are not you joining us ? <sad>
A2: yes
```

```
A1: oh ok
B: sorry
A2: i am so happy
```

* 若A出现两种情感，以后者为准

```
A1: ohh sorry
B: do not worry , you are not the first
A2: glad to hear
```

```
A1: i miss my sister
B: i miss both of you <sad>
A2: 😁
```

* 大部分样本可以仅靠一两个单词判断整段对话的情感

```
A1: Tell me about it
B: your header
A2: 😂
```

```
A1: but i am not one
B: we are all robots .
A2: haha
```

* 少部分样本的标注存在主观性（和个人理解不同）

```
A1: sorry
B: how dare u
A2: i got a new gf
```

## Angry


* 若A，B出现两种情感，以A为准

```
A1: why ?
B: haha you know why ! 😃
A2: your rude i do not want to talk to you
```

```
A1: why you angry me
B: wish i knew <sad>
A2: what you know
```

* 若A出现两种情感，以后者为准

```
A1: i like dominos pizza
B: legit same but there ' s pizza hut
A2: i hate pizza hut
```

* 大部分样本可以仅靠一两个单词判断整段对话的情感

```
A1: cool
B: so what are you doin ?
A2: i m angry on you
```
* 部分样本拼写很不规范

```
A1: watever
B: lol k
A2: i dnt want to tok to u
```

* 有一些情感相对隐晦但对分辨能力的词组

```
A1: get out
B: you got a link ?
A2: do not text me again
```

```
A1: ok give me
B: take your time !
A2: get lost
```

```
A1: no you are pottypiece
B: no u
A2: no you
```

## Sad

* 若A，B出现两种情感，以A为准

```
A1: fine wht abt u
B: i am fine too <happy> how ' s your exams ? <happy>
A2: bad
```

* 若A出现两种情感，以后者为准

```
A1: 😃 yes yes
B: :3 you seem like a happy person
A2: yes 😂 happy outside , 😞 sad inside
```

* 大部分样本可以仅靠一两个单词判断整段对话的情感

```
A1: i feel low
B: why is that
A2: sad life
```

```
A1: u are doing what ???
B: nd u
A2: alone
```

* 少部分样本的情感依賴外面知识 (场景相关)

```
A1: not fine
B: why ? :'o
A2: tomorrow lab exam
```

```
A1: ok
B: hello ! how are you ?
A2: battery low my phon
```

```
A1: i broke up with my gf
B: thats just mean
A2: last year
```

* 意思隱晦

```
A1: bad
B: why ?
A2: because of u
```

## Others 

* 沒有明显情感

```
A1: so teach me
B: i am the student .
A2: realky
```

```
A1: not sure
B: let me know !
A2: pls you tell
```

* 与B的情感无关

```
A1: send me your nekde pic
B: after this battle , sure :D
A2: let us start
```

* 少部分样本的标注存在主观性（和个人理解不同）

```
A1: i like your positive approach
B: no problem , glad to help
A2: 😀 😀
```


