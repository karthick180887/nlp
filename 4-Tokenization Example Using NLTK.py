#!/usr/bin/env python
# coding: utf-8

# In[27]:


corpus = "Hello there! How's your day going? It's amazing. Don't you think so?"


# In[28]:


print(corpus)


# In[29]:


import nltk
nltk.download('punkt_tab')


# In[43]:


##  Tokenization
## Sentence-->paragraphs
from nltk.tokenize import sent_tokenize


# In[44]:


documents=sent_tokenize(corpus)


# In[45]:


type(documents)


# In[46]:


for sentence in documents:
    print(sentence)


# In[47]:


## Tokenization 
## Paragraph-->words
## sentence--->words
from nltk.tokenize import word_tokenize


# In[48]:


word_tokenize(corpus)


# In[37]:


for sentence in documents:
    print(word_tokenize(sentence))


# In[49]:


from nltk.tokenize import wordpunct_tokenize


# In[50]:


wordpunct_tokenize(corpus)


# In[51]:


from nltk.tokenize import TreebankWordTokenizer


# In[52]:


tokenizer=TreebankWordTokenizer()


# In[53]:


tokenizer.tokenize(corpus)


# In[ ]:





# In[ ]:





# In[ ]:




