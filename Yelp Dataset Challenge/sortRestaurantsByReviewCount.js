var fs = require('fs'),
    es = require('event-stream'),
    JSONStream = require('JSONStream')
    _ = require('lodash');
var dict = {
  numRestaurLess50Reviews : 0,
  numRestaurLess100Reviews : 0,
  numRestaurLess150Reviews : 0,
  numRestaurLess200Reviews : 0,
  numRestaurMore200Reviews : 0,
};
var getStream = function () {
    var jsonData = 'yelp_academic_dataset_business.json',
        stream = fs.createReadStream(jsonData, {encoding: 'utf8'}),
        parser = JSONStream.parse();
        return stream.pipe(parser);
}

getStream().pipe(es.mapSync(function (data) {
  //console.log(data['review_count']);
    if(parseInt(data['review_count'])>200) dict['numRestaurMore200Reviews'] += 1;
    else if(parseInt(data['review_count'])<=200) dict['numRestaurLess200Reviews'] += 1;
    if(parseInt(data['review_count'])<=50) dict['numRestaurLess50Reviews'] += 1;
    else if(parseInt(data['review_count'])<=100) dict['numRestaurLess100Reviews'] += 1;
    else if(parseInt(data['review_count'])<=150) dict['numRestaurLess150Reviews'] += 1;
    
    //console.log(arr);
})).on('end', () => {
    console.log(dict);
    console.log('finished');    
});

/*
Output:
{ numRestaurLess50Reviews: 163940,
  numRestaurLess100Reviews: 12739,
  numRestaurLess150Reviews: 4653,
  numRestaurLess200Reviews: 2283,
  numRestaurMore200Reviews: 4978 
  }
*/



