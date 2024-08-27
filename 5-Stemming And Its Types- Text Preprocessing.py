#!/usr/bin/env python
# coding: utf-8

# ## Stemming
# Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as a lemma. Stemming is important in natural language understanding (NLU) and natural language processing (NLP).

# In[1]:


## Classification Problem
## Comments of product is a positive review or negative review
## Reviews----> eating, eat,eaten [going,gone,goes]--->go

words=["eating","eats","eaten","writing","writes","programming","programs","history","finally","finalized"]


# ### PorterStemmer

# In[2]:


from nltk.stem import PorterStemmer


# In[3]:


stemming=PorterStemmer()


# In[4]:


for word in words:
    print(word+"---->"+stemming.stem(word))


# In[5]:


stemming.stem('congratulations')


# In[7]:


stemming.stem("sitting")


# ### RegexpStemmer class
# NLTK has RegexpStemmer class with the help of which we can easily implement Regular Expression Stemmer algorithms. It basically takes a single regular expression and removes any prefix or suffix that matches the expression. Let us see an example

# In[5]:


from nltk.stem import RegexpStemmer


# In[6]:


reg_stemmer=RegexpStemmer('ing$|s$|e$|able$', min=4)


# In[7]:


reg_stemmer.stem('eating')


# In[8]:


reg_stemmer.stem('ingeating')


# In[ ]:





# ### Snowball Stemmer
#  It is a stemming algorithm which is also known as the Porter2 stemming algorithm as it is a better version of the Porter Stemmer since some issues of it were fixed in this stemmer.

# In[10]:


from nltk.stem import SnowballStemmer


# In[11]:


snowballsstemmer=SnowballStemmer('english')


# In[12]:


for word in words:
    print(word+"---->"+snowballsstemmer.stem(word))


# In[13]:


stemming.stem("fairly"),stemming.stem("sportingly")


# In[14]:


snowballsstemmer.stem("fairly"),snowballsstemmer.stem("sportingly")


# In[15]:


snowballsstemmer.stem('goes')


# In[34]:


stemming.stem('goes')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




