import { config } from 'dotenv';
import { join } from 'path';
import { Kontext } from '@kontext.dev/kontext-sdk';
import https from 'https';

// Disable SSL verification for staging API (ONLY FOR TESTING)
const httpsAgent = new https.Agent({
  rejectUnauthorized: false
});

// Patch global fetch to use our custom agent
const originalFetch = global.fetch;
global.fetch = (url, options = {}) => {
  return originalFetch(url, {
    ...options,
    agent: httpsAgent
  });
};

// Load environment variables from .env
config({ path: join(process.cwd(), '.env') });

// Initialize Kontext
const userId = 'd7d3cb73-edf1-46c8-84c4-72703585fdd5';

// Remove trailing slash from API URL if present
let apiUrl = process.env.KONTEXT_API_URL;
if (apiUrl) {
  apiUrl = apiUrl.replace(/\/$/, '');
}

const kontext = new Kontext({ 
  apiKey: process.env.KONTEXT_API_KEY, 
  userId: userId,
  ...(apiUrl ? { apiUrl } : {})
});

async function simpleTest() {
  console.log('🧪 Simple Kontext Test\n');
  console.log('User ID:', userId);
  console.log('API URL:', (process.env.KONTEXT_API_URL || 'default (production)').replace(/\/$/, ''));
  console.log('API Key:', process.env.KONTEXT_API_KEY ? `${process.env.KONTEXT_API_KEY.substring(0, 10)}...` : 'NOT SET');
  console.log('─'.repeat(60));
  console.log('');

  try {
    // Query the vault for supplier data
    console.log('📊 Querying Kontext vault...\n');
    
    const question = 'Which supplier has the highest quality rating?';
    console.log('Question:', question);
    console.log('');
    
    const vaultResult = await kontext.vault.query({
      userId: userId,
      query: question,
      includeAnswer: true,
      topK: 5,
    });

    console.log('✅ Vault Response:');
    console.log('Answer:', vaultResult.answer?.text || 'No answer generated');
    console.log('Documents found:', vaultResult.hits?.length || 0);
    console.log('');

    // Show documents if found
    if (vaultResult.hits && vaultResult.hits.length > 0) {
      console.log('� Documents:');
      vaultResult.hits.slice(0, 3).forEach((hit, i) => {
        const text = hit.attributes?.text || hit.attributes?.snippet || JSON.stringify(hit.attributes);
        console.log(`\nDocument ${i + 1}:`);
        console.log(text.substring(0, 200) + (text.length > 200 ? '...' : ''));
      });
      console.log('');
    }

    console.log('─'.repeat(60));
    console.log('\n✅ Test completed successfully!\n');

  } catch (error) {
    console.error('❌ Error:', error.code || error.name);
    console.error('Message:', error.message);
    
    // More detailed error logging
    if (error.cause) {
      console.error('Cause:', error.cause);
    }
    
    if (error.response) {
      console.error('Status:', error.response.status);
      console.error('Data:', JSON.stringify(error.response.data, null, 2));
    }
    
    if (error.request) {
      console.error('Request URL:', error.request.url || 'N/A');
    }
    
    console.log('\n💡 Troubleshooting:');
    if (error.code === 'TOKEN_ISSUE_FAILED' || error.statusCode === 401) {
      console.log('• API authentication failed - check KONTEXT_API_KEY');
      console.log('• Make sure the API key is valid and active');
    } else if (error.statusCode === 404 || error.message?.includes('404')) {
      console.log('• Vault may be empty - upload data through the web UI');
      console.log('• Or the API endpoint/user ID may be incorrect');
    } else {
      console.log('• Check that data is uploaded to Kontext vault');
      console.log('• Verify KONTEXT_API_KEY is set correctly');
      console.log('• Try uploading data through the web UI first');
    }
    console.log('');
  }
}

// Run the test
console.log('Starting test...\n');
simpleTest();
