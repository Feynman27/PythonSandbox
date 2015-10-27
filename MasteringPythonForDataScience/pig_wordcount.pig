data = load '/tmp/moby_dick/';

word_token = foreach data generate flatten(TOKENIZE((chararray)$0)) as word;

group_word_token = group word_token by word;

count_word_token = foreach group_word_token generate COUNT(word_token) as cnt, group;

sort_word_token = ORDER count_word_token by cnt DESC;

top10_word_count = LIMIT sort_word_token 10;

DUMP top10_word_count;
