import requests
from bs4 import BeautifulSoup
import os
import re

#USER INPUT===========================
GENIUS_API_TOKEN='INSERT TOKEN HERE'
target_artist = 'INSERT ARTIST NAME'
target_amount = 30 #INSERT SONG NUMBER
#=====================================

# get artist from genius via api
def request_artist_info(artist_name, page):
	base_url = 'https://api.genius.com'
	# who's accessing? me lol
	headers = {'Authorization': 'Bearer ' + GENIUS_API_TOKEN}
	# search base url
	search_url = base_url + '/search?per_page=10&page=' + str(page)
	data = {'q': artist_name}
	response = requests.get(search_url, params=data, headers=headers)
	return response

# get song urls from artist grabbed above
def request_song_url(artist_name,song_cap):
	page = 1
	songs = []

	while True:
		response = request_artist_info(artist_name, page)
		response.raise_for_status()  # raises exception when not a 2xx response
		if response.status_code != 204:
			response = response.json()

		if 'response' in json and 'hits' in json['response']:
			song_info = []
			for hit in json['response']['hits']:
				if artist_name.lower() in hit['result']['primary_artist']['name'].lower():
					song_info.append(hit)
		else:
			break

		# collect songs from songs structure
		for song in song_info:
			if (len(songs) < song_cap):
				url = song['result']['url']
				songs.append(url)
		if (len(songs)==song_cap):
			break
		else:
			page += 1
	print('Found {} songs by {}'.format(len(songs), artist_name))
	return songs

# scrape lyrics from genius.com song url
def scrape_song_lyrics(url):
	page = requests.get(url)
	html = BeautifulSoup(page.text, 'html.parser')
	lyrics = html.find('div', class_='lyrics').get_text()
	lyrics = re.sub(r'[\(\[].*?[\)\]]', '',lyrics)
	lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
	return lyrics

# loop through urls gathered and direct lyrics to a single txt file
def write_lyrics_to_file(artist_name, song_count):
    f = open(artist_name.lower() + '.txt', 'w')
    urls = request_song_url(artist_name, song_count)
    for url in urls:
        lyrics = scrape_song_lyrics(url)
        f.write(lyrics.encode("utf8"))
    f.close()
    num_lines = sum(1 for line in open(artist_name.lower() + '.txt', 'rb'))
    print('Wrote {} lines to file from {} songs'.format(num_lines, song_count))

#grab lyrics and write to file
write_lyrics_to_file(target_artist,target_amount)
