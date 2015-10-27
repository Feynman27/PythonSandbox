/*
Determine sentiment score for jurassic world using UDF in pig
*/

register 'positive_sentiment.py' using org.apache.pig.scripting.jython.JythonScriptEngine as positive;

register 'negative_sentiment.py' using org.apache.pig.scripting.jython.JythonScriptEngine as negative;

data = load '/tmp/jurassic_world/';

feedback_sentiments = foreach data generate LOWER((chararray)$0) as feedback, positive.sentiment_score(LOWER((chararray)$0)) as psenti, negative.sentiment_score(LOWER((chararray)$0)) as nsenti;

average_sentiments = foreach feedback_sentiments generate feedback, psenti+nsenti;

dump average_sentiments;
